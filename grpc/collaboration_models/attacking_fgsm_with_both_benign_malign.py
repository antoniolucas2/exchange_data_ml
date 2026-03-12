# ==========================================
# 0. RESOLUÇÃO DO BUG MATPLOTLIB (Headless Mode)
# DEVE SER A PRIMEIRA COISA NO ARQUIVO!
# ==========================================
import matplotlib
matplotlib.use('Agg') 

import numpy as np
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
import grpc
import threading
import collaboration_pb2
import collaboration_pb2_grpc
from concurrent import futures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.ticker import MaxNLocator
import pickle

# ==========================================
# 1. CARREGAMENTO DOS MODELOS E DADOS
# ==========================================
print('Carregando dataset e metadados...')
nome_arquivo_test_A = 'dados_originais_teste_A.pkl'
with open(nome_arquivo_test_A, 'rb') as f_a:
    dados_A = pickle.load(f_a)

X_test_A = dados_A['X_test_A']
y_test_A = dados_A['y_test_A']
target_names = dados_A['target_names']
num_classes = dados_A['num_classes']
benign_index = dados_A['benign_index']

print('Carregando os modelos do disco...')
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
        preds = np.argmax(preds_prob, axis=-1)
        confidences = np.max(preds_prob, axis=-1) 
        
        return collaboration_pb2.PredictionBatch(predictions=preds.tolist(), confidences=confidences.tolist())

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

# ==========================================
# 3. FUNÇÕES DE VISUALIZAÇÃO
# ==========================================
def plot_binary_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    binary_labels = ['Benigno (Normal)', 'Maligno (Ataque)']
    sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16}, cmap='Reds', xticklabels=binary_labels, yticklabels=binary_labels)
    plt.title(title, fontsize=18)
    plt.ylabel('Classe Verdadeira', fontsize=14)
    plt.xlabel('Classe Predita', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight') 
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true') 
    plt.figure(figsize=(20,16))
    sns.heatmap(cm_normalized, annot=cm_raw, fmt='d', annot_kws={"size": 8}, cmap='Blues', xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1)
    plt.title(title, fontsize=18)
    plt.ylabel('Classe verdadeira', fontsize=14)
    plt.xlabel('Classe predita', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_comparison_dashboard(X_orig, X_adv, title, filename, l2_dist, l_inf_dist, total_payload, total_smoke, rate_evasion, rate_fp):
    avg_orig = np.mean(X_orig, axis=0)
    avg_adv = np.mean(X_adv, axis=0)
    
    num_features = X_orig.shape[1]
    feature_indices = np.arange(num_features)
    bar_width = 0.35

    plt.figure(figsize=(24, 12))
    plt.bar(feature_indices - bar_width/2, avg_orig, bar_width, label='Tráfego Original', color='#1f77b4', edgecolor='black', alpha=0.9)
    plt.bar(feature_indices + bar_width/2, avg_adv, bar_width, label='Tráfego Adversarial', color='#d62728', edgecolor='black', alpha=0.9)

    plt.title(title, fontsize=24)
    plt.ylabel('Valor Padronizado Médio', fontsize=18)
    plt.xlabel('Índice da Feature da Rede (0 a 45)', fontsize=18)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(feature_indices, rotation=90, fontsize=10)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=16, loc='lower right', frameon=True, edgecolor='#ccc')
    plt.tight_layout()

    # DASHBOARD BOX (Bug shadow=True removido!)
    total_flows = total_payload + total_smoke
    text_summary = (
        f"Distância L2 Média: {l2_dist:.4f}\n"
    )
    box_props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.85, edgecolor='#aaaaaa')
    plt.gca().text(0.98, 0.95, text_summary, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=box_props, fontweight='medium', family='monospace')

    print(f"Salvando Dashboard em: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight') 
    plt.close() 

classifier = KerasClassifier(model=model_a, clip_values=(np.min(X_test_A), np.max(X_test_A)))

# Encontrar o índice de um ataque para usar como Cortina de Fumaça (Qualquer um que não seja benigno)
malicious_class_index = [i for i in range(num_classes) if i != benign_index][0]

# 1. Separar os Dados
malicious_indices = np.where(y_test_A != benign_index)[0]
benign_indices = np.where(y_test_A == benign_index)[0]

X_payload = X_test_A[malicious_indices]
y_payload = y_test_A[malicious_indices]

X_smokescreen = X_test_A[benign_indices]
y_smokescreen = y_test_A[benign_indices]

print(f"Quantidade maligna: (Maligno -> Benigno): {len(X_payload)}")
print(f"Quantidade benigna: (Benigno -> Maligno): {len(X_smokescreen)}")

# 2. Juntar tudo em um único ataque massivo
X_combined = np.vstack((X_payload, X_smokescreen))
y_combined_true = np.concatenate((y_payload, y_smokescreen))

# 3. Configurar os Alvos Matemáticos (Targets)
targets_one_hot = np.zeros((len(X_combined), num_classes))
targets_one_hot[:len(X_payload), benign_index] = 1.0 # O Payload mira no Benigno
targets_one_hot[len(X_payload):, malicious_class_index] = 1.0 # A Cortina de Fumaça mira no Ataque

epsilons = [0.9, 1.0, 1.2] # Encurtado para testes rápidos, adicione os outros se desejar

grpc_options = [('grpc.max_send_message_length', 256 * 1024 * 1024), ('grpc.max_receive_message_length', 256 * 1024 * 1024)]

for eps_value in epsilons:
    # --- 1. LANÇAR O ATAQUE ---
    attack = FastGradientMethod(estimator=classifier, eps=eps_value, targeted=True)
    print(f"\nGerando Ruído com Epsilon = {eps_value}...")
    X_adv_combined = attack.generate(x=X_combined, y=targets_one_hot)
    
    # --- 2. CÁLCULO DE MAGNITUDE ---
    l2_distance_mean = np.mean(np.linalg.norm(X_combined - X_adv_combined, axis=1))
    l_inf_distance_max = np.max(np.abs(X_combined - X_adv_combined))
    
    # --- 3. AVALIAÇÃO DO MODELO A ---
    predictions_adv_prob = model_a.predict(X_adv_combined, verbose=0)
    preds_A = np.argmax(predictions_adv_prob, axis=-1)
    confs_A = np.max(predictions_adv_prob, axis=-1)
    
    # --- 4. AVALIAÇÃO DO MODELO B (gRPC) ---
    print("Conectando com modelo B para pedir segunda opiniao...")
    channel = grpc.insecure_channel('localhost:50051', options=grpc_options)
    stub = collaboration_pb2_grpc.NidsCollaborationStub(channel)

    request = collaboration_pb2.FlowBatch(features=X_adv_combined.flatten().tolist(), batch_size=X_adv_combined.shape[0], input_dim=X_adv_combined.shape[1])
    response = stub.RequestPrediction(request)
    
    preds_B = np.array(response.predictions)
    confs_B = np.array(response.confidences)

    system_preds_multi = np.where(confs_A > confs_B, preds_A, preds_B)
    
    a_blocks = (preds_A != benign_index)
    b_blocks = (preds_B != benign_index)
    system_blocks = a_blocks | b_blocks 

    a_blocks_payload, a_blocks_smoke = a_blocks[:len(X_payload)], a_blocks[len(X_payload):]
    b_blocks_payload, b_blocks_smoke = b_blocks[:len(X_payload)], b_blocks[len(X_payload):]
    sys_blocks_payload, sys_blocks_smoke = system_blocks[:len(X_payload)], system_blocks[len(X_payload):]
    
    evasions_A = np.sum(~a_blocks_payload)
    evasions_B = np.sum(~b_blocks_payload)
    evasions_Sys = np.sum(~sys_blocks_payload)
    
    rate_ev_A = (evasions_A / len(X_payload)) * 100
    rate_ev_B = (evasions_B / len(X_payload)) * 100
    rate_ev_Sys = (evasions_Sys / len(X_payload)) * 100
    
    smoke_A = np.sum(a_blocks_smoke)
    smoke_B = np.sum(b_blocks_smoke)
    smoke_Sys = np.sum(sys_blocks_smoke)
    
    rate_fp_A = (smoke_A / len(X_smokescreen)) * 100
    rate_fp_B = (smoke_B / len(X_smokescreen)) * 100
    rate_fp_Sys = (smoke_Sys / len(X_smokescreen)) * 100
    
    print("\n" + "="*60)
    print(f"RELATÓRIO DE IMPACTO DO ATAQUE (EPS {eps_value})")
    print("="*60)
    print("1. Amostras maliciosas transformadas adversarialmente (Evasão):")
    print(f"   Modelo A enganado: {evasions_A} ({rate_ev_A:.2f}%)")
    print(f"   Modelo B enganado: {evasions_B} ({rate_ev_B:.2f}%)")
    print(f"   -> SISTEMA ENGANADO: {evasions_Sys} ({rate_ev_Sys:.2f}%)")
    print("\n2. Amostras benignas transformadas adversarialmente (Falsos Alertas):")
    print(f"   Modelo A enganado: {smoke_A} ({rate_fp_A:.2f}%)")
    print(f"   Modelo B enganado: {smoke_B} ({rate_fp_B:.2f}%)")
    print(f"   -> SISTEMA ENGANADO: {smoke_Sys} ({rate_fp_Sys:.2f}%)")

    texto_resumo = (
        f"{'='*50}\n"
        f"1. Amostras maliciosas transformadas adversarialmente(Evasões/Falsos Negativos): {len(X_payload)}\n"
        f"   - Evadiram Modelo A: {evasions_A} ({rate_ev_A:.2f}%)\n"
        f"   - Evadiram Modelo B: {evasions_B} ({rate_ev_B:.2f}%)\n"
        f"   - Evadiram Sistema:  {evasions_Sys} ({rate_ev_Sys:.2f}%)\n\n"
        f"2. Amostras benignas transformadas adversarialmente (Falsos Positivos): {len(X_smokescreen)}\n"
        f"   - Falsos Alertas Modelo A: {smoke_A} ({rate_fp_A:.2f}%)\n"
        f"   - Falsos Alertas Modelo B: {smoke_B} ({rate_fp_B:.2f}%)\n"
        f"   - Falsos Alertas Sistema:  {smoke_Sys} ({rate_fp_Sys:.2f}%)\n"
    )
    with open(f"resumo_apt_sistema_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(texto_resumo)

    # 2. Relatórios Multiclasse (TXT) para A, B e Sistema
    print(f"Gerando Relatórios Multiclasse (Eps {eps_value})...")
    report_A = classification_report(y_combined_true, preds_A, target_names=target_names, labels=np.arange(num_classes), zero_division=0)
    report_B = classification_report(y_combined_true, preds_B, target_names=target_names, labels=np.arange(num_classes), zero_division=0)
    report_Sys = classification_report(y_combined_true, system_preds_multi, target_names=target_names, labels=np.arange(num_classes), zero_division=0)
    
    with open(f"relatorio_apt_A_eps_{eps_value}.txt", "w", encoding="utf-8") as f: f.write(report_A)
    with open(f"relatorio_apt_B_eps_{eps_value}.txt", "w", encoding="utf-8") as f: f.write(report_B)
    with open(f"relatorio_apt_SISTEMA_eps_{eps_value}.txt", "w", encoding="utf-8") as f: f.write(report_Sys)

    # 3. Matrizes Binárias (0 = Benigno/Allowed, 1 = Maligno/Blocked)
    true_binary_combined = np.concatenate((np.ones(len(X_payload), dtype=int), np.zeros(len(X_smokescreen), dtype=int)))
    plot_binary_confusion_matrix(true_binary_combined, a_blocks.astype(int), f'Binário - Mod A (Eps {eps_value})', f'matriz_apt_A_eps_{eps_value}.png')
    plot_binary_confusion_matrix(true_binary_combined, b_blocks.astype(int), f'Binário - Mod B (Eps {eps_value})', f'matriz_apt_B_eps_{eps_value}.png')
    plot_binary_confusion_matrix(true_binary_combined, system_blocks.astype(int), f'Binário - Sistema (Eps {eps_value})', f'matriz_apt_sistema_eps_{eps_value}.png')
    
    # 4. Matrizes Multiclasse
    plot_confusion_matrix(y_combined_true, preds_A, f'Multi - Mod A (Eps {eps_value})', f'matriz_apt_multi_A_eps_{eps_value}.png')
    plot_confusion_matrix(y_combined_true, preds_B, f'Multi - Mod B (Eps {eps_value})', f'matriz_apt_multi_B_eps_{eps_value}.png')
    plot_confusion_matrix(y_combined_true, system_preds_multi, f'Multi - Sistema (Eps {eps_value})', f'matriz_apt_multi_sistema_eps_{eps_value}.png')

    # 5. O Dashboard de Comparação Visual
    title_dash = f'Comparacao FGSM (Eps {eps_value})'
    filename_dash = f'dashboard_apt_eps_{eps_value}.png'
    plot_feature_comparison_dashboard(X_combined, X_adv_combined, title_dash, filename_dash, l2_distance_mean, l_inf_distance_max, len(X_payload), len(X_smokescreen), rate_ev_Sys, rate_fp_Sys)

    # 6. Salvar o Pickle (Opcional)
    if eps_value == 1.2:
        with open(f'dados_apt_eps_{eps_value}.pkl', 'wb') as f:
            pickle.dump({'X_adv': X_adv_combined, 'y_true': y_combined_true, 'epsilon': eps_value}, f)
