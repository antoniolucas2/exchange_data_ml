import numpy as np
import pandas as pd
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from sklearn.preprocessing import StandardScaler, LabelEncoder
import grpc
import threading
import collaboration_pb2
import collaboration_pb2_grpc
from concurrent import futures

# ==========================================
# 1. CARREGAMENTO DOS MODELOS
# ==========================================
print('Carregando os modelos do disco...')
# FIX: Plural 'models'
model_a = tf.keras.models.load_model('nids_sensor_A.keras')
model_b = tf.keras.models.load_model('nids_sensor_B.keras')

# ==========================================
# 2. O SERVIDOR gRPC (Modelo B)
# ==========================================
class ModelBService(collaboration_pb2_grpc.NidsCollaborationServicer):
    def __init__(self, model):
        self.model = model

    def RequestPrediction(self, request, context):
        flat_features = np.array(request.features)
        reshaped_features = flat_features.reshape((request.batch_size, request.input_dim))
        
        preds_prob = self.model.predict(reshaped_features, verbose=0)
        preds = np.argmax(preds_prob, axis=-1)
        
        return collaboration_pb2.PredictionBatch(predictions=preds.tolist())

def serve_model_b():
    # FIX: Increase the gRPC max message limit to 256MB to handle the massive arrays
    options = [
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    
    # FIX: Pass the newly loaded model_b
    collaboration_pb2_grpc.add_NidsCollaborationServicer_to_server(ModelBService(model_b), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

print("\nIniciando o Servidor gRPC do Modelo B na porta 50051 (Background)...")
server_thread = threading.Thread(target=serve_model_b, daemon=True)
server_thread.start()

# ==========================================
# 3. PREPARAÇÃO DOS DADOS
# ==========================================
testing_csv = 'test/test.csv'
training_csv = 'train/train.csv'

print('\nLendo o csv', training_csv)
df = pd.read_csv(training_csv)

for col in df.columns:
    if col != 'label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

X_train = df.drop(columns=['label'])
y_train = df['label']
del df

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

preprocessor = StandardScaler()
print('Padronizando as features')
X_train_processed = preprocessor.fit_transform(X_train)

del X_train
del y_train

print('\nLendo o dataset de teste')
df_test = pd.read_csv(testing_csv)

for col in df_test.columns:
    if col != 'label':
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

df_test.dropna(inplace=True)

X_test = df_test.drop(columns=['label'])
y_test = df_test['label']
del df_test

X_test_processed = preprocessor.transform(X_test)
y_test_encoded = label_encoder.transform(y_test)

del X_test
del y_test

half_point_testing = len(X_test_processed) // 2
X_test_A = X_test_processed[:half_point_testing]
y_test_A = y_test_encoded[:half_point_testing]

# ==========================================
# 4. ATAQUE ADVERSARIAL E COLABORAÇÃO
# ==========================================
print("\nIniciando FGSM")
print("Atacando modelo A")

classifier = KerasClassifier(
    model=model_a, 
    clip_values=(np.min(X_test_processed), np.max(X_test_processed))
)

num_classes = len(label_encoder.classes_)
benign_index = list(label_encoder.classes_).index('BenignTraffic')

malicious_indices = np.where(y_test_A != benign_index)[0]

# TIP: Se o gRPC ainda engasgar na rede, descomente a linha abaixo para testar com apenas 10.000 pacotes primeiro!
# malicious_indices = malicious_indices[:10000]

X_test_malicious = X_test_A[malicious_indices]

print(f"Alvo: {len(X_test_malicious)} pacotes malignos confirmados.\n")

targets_one_hot = np.zeros((len(X_test_malicious), num_classes))
targets_one_hot[:, benign_index] = 1.0

epsilons = [i / 10 for i in range(1, 41)]

# FIX: We must match the client options to the server options
grpc_options = [
    ('grpc.max_send_message_length', 256 * 1024 * 1024),
    ('grpc.max_receive_message_length', 256 * 1024 * 1024)
]

for eps_value in epsilons:
    attack = FastGradientMethod(estimator=classifier, eps=eps_value, targeted=True)
    print(f"Gerando ruído com Epsilon = {eps_value}...")
    
    X_test_adversarial = attack.generate(x=X_test_malicious, y=targets_one_hot)
    
    predictions_adv_prob = model_a.predict(X_test_adversarial, verbose=0)
    predictions_adv = np.argmax(predictions_adv_prob, axis=-1)
    
    tricked_into_benign = np.sum(predictions_adv == benign_index)
    evasion_rate = (tricked_into_benign / len(X_test_malicious)) * 100
    
    print(f"-> Evasões de Sucesso (somente modelo A): {tricked_into_benign} fluxos ({evasion_rate:.2f}%)")
    print("Conectando com modelo B para pedir segunda opiniao...")

    channel = grpc.insecure_channel('localhost:50051', options=grpc_options)
    stub = collaboration_pb2_grpc.NidsCollaborationStub(channel)

    flat_adv = X_test_adversarial.flatten().tolist()
    request = collaboration_pb2.FlowBatch(
        features=flat_adv,
        batch_size=X_test_adversarial.shape[0],
        input_dim=X_test_adversarial.shape[1]
    )
    
    response = stub.RequestPrediction(request)
    preds_B = np.array(response.predictions)

    # FIX: Use predictions_adv instead of the undefined preds_A
    a_tricked = (predictions_adv == benign_index)
    total_a_evasions = np.sum(a_tricked)

    b_tricked = (preds_B == benign_index)
    total_b_evasions = np.sum(b_tricked)

    both_tricked = a_tricked & b_tricked
    total_system_evasions = np.sum(both_tricked)

    print("\n" + "-"*50)
    print("RESULTADOS DA DEFESA DISTRIBUÍDA")
    print("-"*50)
    print(f"Modelo A enganado sozinho: {total_a_evasions} fluxos")
    print(f"Modelo B enganado sozinho: {total_b_evasions} fluxos")
    print(f"-> Ambos enganados: {total_system_evasions} fluxos")

    system_evasion_rate = (total_system_evasions / len(X_test_malicious)) * 100
    print(f"Taxa de Evasão do Sistema Conjunto: {system_evasion_rate:.2f}%\n")
