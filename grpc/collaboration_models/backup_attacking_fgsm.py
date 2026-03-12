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

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix

import pickle
from sklearn.metrics import classification_report

# ==========================================
# 1. CARREGAMENTO DOS MODELOS
# ==========================================
print('Carregando os modelos do disco...')
# FIX: Plural 'models'
model_a = tf.keras.models.load_model('nids_sensor_A_versao_3.keras')
model_b = tf.keras.models.load_model('nids_sensor_B_versao_3.keras')

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
        
        # Obter a classe predita E a probabilidade máxima (certeza)
        preds = np.argmax(preds_prob, axis=-1)
        confidences = np.max(preds_prob, axis=-1) 
        
        return collaboration_pb2.PredictionBatch(
            predictions=preds.tolist(), 
            confidences=confidences.tolist()
        )
def serve_model_b():
    options = [
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    
    collaboration_pb2_grpc.add_NidsCollaborationServicer_to_server(ModelBService(model_b), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

print("\nIniciando o Servidor gRPC do Modelo B na porta 50051 (Background)...")
server_thread = threading.Thread(target=serve_model_b, daemon=True)
server_thread.start()

def plot_binary_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))

    binary_labels = ['Benigno (Normal)', 'Maligno (Ataque)']

    sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16}, cmap='Reds', 
                xticklabels=binary_labels, yticklabels=binary_labels)

    plt.title(title, fontsize=18)
    plt.ylabel('Classe Verdadeira', fontsize=14)
    plt.xlabel('Classe Predita', fontsize=14)
    plt.tight_layout()
    
    print(f"Salvando matriz binária em: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight') 
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20,16))

    sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 8}, cmap='Blues', xticklabels=target_names, yticklabels=target_names)

    plt.title(title, fontsize=18)
    plt.ylabel('Classe verdadeira', fontsize=14)
    plt.xlabel('Classe predita', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. PREPARAÇÃO DOS DADOS
# ==========================================

nome_arquivo_test_A = 'dados_originais_teste_A.pkl'

with open(nome_arquivo_test_A, 'rb') as f_a:
    dados_A = pickle.load(f_a)

X_test_A = dados_A['X_test_A']
y_test_A = dados_A['y_test_A']
target_names = dados_A['target_names']
num_classes = dados_A['num_classes']
benign_index = dados_A['benign_index']

# ==========================================
# 4. ATAQUE ADVERSARIAL E COLABORAÇÃO
# ==========================================
print("\nIniciando FGSM")
print("Atacando modelo A")

classifier = KerasClassifier(
    model=model_a, 
    clip_values=(np.min(X_test_A), np.max(X_test_A))
)

malicious_indices = np.where(y_test_A != benign_index)[0]

X_test_malicious = X_test_A[malicious_indices]
y_test_malicious = y_test_A[malicious_indices]

print(f"Alvo: {len(X_test_malicious)} pacotes malignos confirmados.\n")

targets_one_hot = np.zeros((len(X_test_malicious), num_classes))
targets_one_hot[:, benign_index] = 1.0

epsilons = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

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
    confs_B = np.array(response.confidences)

    confs_A = np.max(predictions_adv_prob, axis=-1)

    system_final_preds = np.where(confs_A > confs_B, predictions_adv, preds_B)

    print(f"\n--- MÉTRICAS DO SISTEMA CONJUNTO (MAX CONFIDENCE - Eps {eps_value}) ---")
    report_system = classification_report(y_test_malicious, system_final_preds, 
                                          target_names=target_names, 
                                          labels=np.arange(num_classes), 
                                          zero_division=0)
    print(report_system)

    # Salvar o relatório do sistema em TXT
    with open(f"relatorio_adv_SISTEMA_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(report_system)

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

    true_adv_binary = np.ones(len(X_test_malicious), dtype=int)
    
    # 2. Convert Model A and Model B's adversarial predictions to Binary
    # If the prediction is the benign_index (0), it's a 0. Otherwise, it's a 1.
    preds_A_adv_binary = (predictions_adv != benign_index).astype(int)
    preds_B_adv_binary = (preds_B != benign_index).astype(int)
    
    # 3. Plot and save Model A's matrix
    title_A = f'Matriz FGSM - Modelo A (Eps {eps_value})'
    filename_A = f'matriz_fgsm_A_eps_{eps_value}.png'
    plot_binary_confusion_matrix(true_adv_binary, preds_A_adv_binary, title_A, filename_A)
    
    # 4. Plot and save Model B's matrix
    title_B = f'Matriz FGSM - Modelo B (Eps {eps_value})'
    filename_B = f'matriz_fgsm_B_eps_{eps_value}.png'
    plot_binary_confusion_matrix(true_adv_binary, preds_B_adv_binary, title_B, filename_B)

    title_multi_A = f'FGSM Multiclasse - Modelo A (Eps {eps_value})'
    filename_multi_A = f'matriz_fgsm_multi_A_eps_{eps_value}.png'
    plot_confusion_matrix(y_test_malicious, predictions_adv, title_multi_A, filename_multi_A)
    
    title_multi_B = f'FGSM Multiclasse - Modelo B (Eps {eps_value})'
    filename_multi_B = f'matriz_fgsm_multi_B_eps_{eps_value}.png'
    plot_confusion_matrix(y_test_malicious, preds_B, title_multi_B, filename_multi_B)

    # 2. Matriz Binária para o Sistema Conjunto (2x2)
    # A matriz verdadeira é composta apenas de ataques (Tudo 1)
    true_adv_binary = np.ones(len(X_test_malicious), dtype=int)
    
    # Lógica do Ensemble: Se ambos foram enganados (both_tricked = True), a predição final é 0 (Benigno).
    # Caso contrário (~ invertido), o sistema pegou o ataque e a predição é 1 (Maligno).
    preds_system_binary = (~both_tricked).astype(int)
    
    title_system = f'FGSM Binário - Sistema Conjunto (Eps {eps_value})'
    filename_system = f'matriz_fgsm_sistema_eps_{eps_value}.png'
    plot_binary_confusion_matrix(true_adv_binary, preds_system_binary, title_system, filename_system)

    print(f"\nGerando Relatório de Classificação Multiclasse (Eps {eps_value})...")
    
    # zero_division=0 prevents warnings when a specific attack class has 0 successful predictions
    report_adv_A = classification_report(y_test_malicious, predictions_adv, 
                                         target_names=target_names, 
                                         labels=np.arange(num_classes), 
                                         zero_division=0)
    
    report_adv_B = classification_report(y_test_malicious, preds_B, 
                                         target_names=target_names, 
                                         labels=np.arange(num_classes), 
                                         zero_division=0)
    
    print(f"\n--- MÉTRICAS DO MODELO A (Eps {eps_value}) ---")
    print(report_adv_A)
    
    print(f"\n--- MÉTRICAS DO MODELO B (Eps {eps_value}) ---")
    print(report_adv_B)

    # Se quiser salvar o relatório em TXT também:
    with open(f"relatorio_adv_A_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(report_adv_A)
    with open(f"relatorio_adv_B_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(report_adv_B)

    pickle_filename = f'dados_adversariais_eps_{eps_value}.pkl'
    print(f"Salvando flows adversariais em disco: {pickle_filename}")
    
    # Criamos um dicionário para manter os dados amarrados
    adversarial_dict = {
        'X_adv': X_test_adversarial,
        'y_true': y_test_malicious,
        'epsilon': eps_value
    }
    
    if eps_value == 1.2:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(adversarial_dict, f)
