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

model_a = tf.keras.models.load_model('nids_sensor_A.keras')
model_b = tf.keras.models.load_model('nids_sensor_B.keras')

class ModelBService(collaboration_pb2_grpc.NidsCollaborationServicer):
    def __init__(self, model):
        self.model = model

    def RequestPrediction(self, request, context):
        # 1. Reconstruct the 2D array from the incoming 1D list
        flat_features = np.array(request.features)
        reshaped_features = flat_features.reshape((request.batch_size, request.input_dim))
        
        # 2. Model B makes its predictions
        preds_prob = self.model.predict(reshaped_features, verbose=0)
        preds = np.argmax(preds_prob, axis=-1)
        
        # 3. Send the predictions back to Model A
        return collaboration_pb2.PredictionBatch(predictions=preds.tolist())

def serve_model_b():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # We pass loaded_model_b to the service
    collaboration_pb2_grpc.add_NidsCollaborationServicer_to_server(ModelBService(model_b), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

print("\nIniciando o Servidor gRPC do Modelo B na porta 50051 (Background)...")
server_thread = threading.Thread(target=serve_model_b, daemon=True)
server_thread.start()

testing_csv = 'test/test.csv'
training_csv = 'train/train.csv'

print('Lendo o csv', training_csv)
df = pd.read_csv(training_csv)

print('Deixando as colunas apenas com numeros')
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

print('Lendo o dataset de teste')
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

print('Fase de testes para os modelos A e B')

half_point_testing = len(X_test_processed) // 2

X_test_A = X_test_processed[:half_point_testing]
y_test_A = y_test_encoded[:half_point_testing]

X_test_B = X_test_processed[half_point_testing:]
y_test_B = y_test_encoded[half_point_testing:]

print("Iniciando FGSM\n")

print("Atacando modelo A")

# 1. Wrap the model
classifier = KerasClassifier(
    model=model_a, 
    clip_values=(np.min(X_test_processed), np.max(X_test_processed))
)

num_classes = len(label_encoder.classes_)
benign_index = list(label_encoder.classes_).index('BenignTraffic')

# 2. Isolate the malicious packets
malicious_indices = np.where(y_test_A != benign_index)[0]
X_test_malicious = X_test_A[malicious_indices]

print(f"Alvo: {len(X_test_malicious)} pacotes malignos confirmados.\n")

# 3. Create the TARGET array (The Sniper Approach)
# We create a one-hot encoded array that tells ART: "Make ALL of these look like class 'benign_index'"
targets_one_hot = np.zeros((len(X_test_malicious), num_classes))
targets_one_hot[:, benign_index] = 1.0

# 4. Escalate the attack strength
epsilons = [0.1, 0.5, 1.0, 2.0, 3.0]

for eps_value in epsilons:
    # Notice targeted=True. This changes the entire math behind the attack!
    attack = FastGradientMethod(estimator=classifier, eps=eps_value, targeted=True)
    
    print(f"Gerando ruído com Epsilon = {eps_value}...")
    
    # Generate attacks aiming specifically at the Benign class
    X_test_adversarial = attack.generate(x=X_test_malicious, y=targets_one_hot)
    
    # Test the firewall
    predictions_adv_prob = model_a.predict(X_test_adversarial, verbose=0) # verbose=0 keeps terminal clean
    predictions_adv = np.argmax(predictions_adv_prob, axis=-1)
    
    # Count how many perfectly disguised as Benign
    tricked_into_benign = np.sum(predictions_adv == benign_index)
    evasion_rate = (tricked_into_benign / len(X_test_malicious)) * 100
    
    print(f"-> Evasões de Sucesso (somente modelo A): {tricked_into_benign} pacotes ({evasion_rate:.2f}%)\n")

    print("Conectando com modelo B para pedir segunda opiniao")

    channel = grpc.insecure_channel('localhost:50051')
    stub = collaboration_pb2_grpc.NidsCollaborationStub(channel)

    # Flatten the array and pack it into the Protobuf message
    flat_adv = X_test_adversarial.flatten().tolist()
    request = collaboration_pb2.FlowBatch(
        features=flat_adv,
        batch_size=X_test_adversarial.shape[0],
        input_dim=X_test_adversarial.shape[1]
    )
    response = stub.RequestPrediction(request)
    preds_B = np.array(response.predictions)

    a_tricked = (predictions_adv == benign_index)
    total_a_evasions = np.sum(a_tricked)

    # Count how many times B was tricked
    b_tricked = (preds_B == benign_index)
    total_b_evasions = np.sum(b_tricked)

    # Count how many times BOTH were tricked (The True Evasion Rate of the System)
    both_tricked = a_tricked & b_tricked
    total_system_evasions = np.sum(both_tricked)

    print("\n" + "="*50)
    print("RESULTADOS DA DEFESA DISTRIBUÍDA")
    print("="*50)
    print(f"Ataques totais: {len(X_test_malicious)}")
    print(f"Modelo A enganado sozinho: {total_a_evasions} pacotes")
    print(f"Modelo B enganado sozinho: {total_b_evasions} pacotes")
    print(f"-> FALHA CRÍTICA (Ambos enganados): {total_system_evasions} pacotes")

    system_evasion_rate = (total_system_evasions / len(X_test_malicious)) * 100
    print(f"\nTaxa de Evasão do Sistema Conjunto: {system_evasion_rate:.2f}%")

