import numpy as np
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
import grpc
import threading
import collaboration_pb2
import collaboration_pb2_grpc
from concurrent import futures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle

from matplotlib.ticker import MaxNLocator

def plot_feature_comparison_dashboard(X_orig, X_adv, title, filename, target_names, l2_dist, l_inf_dist):
    """
    Creates a grouped bar chart comparing average features (dashboard-style).
    Includes a textbox summarizing the mathematical magnitude of the attack.
    """
    
    # 1. Calculate the mean (average) for every feature (axis=0)
    avg_orig = np.mean(X_orig, axis=0)
    avg_adv = np.mean(X_adv, axis=0)
    
    num_features = X_orig.shape[1]
    feature_indices = np.arange(num_features)
    
    # 2. Setup the massive canvas for tabular data (46 features x 2 bars)
    plt.figure(figsize=(24, 12))
    
    # Increase the distance slightly so they are clearly grouped
    bar_width = 0.35

    # --- Plot the Groups ---
    # Standard Malicious Data (Blue - Standard traffic color)
    plt.bar(feature_indices - bar_width/2, avg_orig, bar_width, label='Original (Maligno)', color='#1f77b4', edgecolor='black', alpha=0.9)
    
    # Adversarial Data (Red - The "hacked" color)
    plt.bar(feature_indices + bar_width/2, avg_adv, bar_width, label='Adversarial (Maligno -> Benigno)', color='#d62728', edgecolor='black', alpha=0.9)

    # --- Formatting the Graph ---
    plt.title(title, fontsize=24)
    plt.ylabel('Valor Padronizado Médio', fontsize=18)
    plt.xlabel('Índice da Feature da Rede (0 a 45)', fontsize=18)
    
    # Use MaxNLocator to force integers (0, 1, 2...) on the axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(feature_indices, rotation=90, fontsize=10) # Heavy rotation for 46 names
    plt.yticks(fontsize=14)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Placing the legend at the bottom to leave the top-right empty for the summary box
    plt.legend(fontsize=16, loc='lower right', frameon=True, shadow=True, edgecolor='#ccc')
    plt.tight_layout()

    # ==========================================
    # --- NOVA CAIXA RESUMO (ESTILO DASHBOARD) ---
    # ==========================================
    
    # Formating the text summary clearly
    text_summary = (
        f"Distância L2 Média por fluxo: {l2_dist:.4f}\n"
    )
    
    # Define a translucent, professional border
    box_props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.85, edgecolor='#aaaaaa')

    # Use 'transform=plt.gca().transAxes' to position based on screen percentage (not data coordinates)
    # Positioning this in the blank top-right area (x=0.98, y=0.95)
    plt.gca().text(0.98, 0.95, text_summary, transform=plt.gca().transAxes, 
                   fontsize=14, verticalalignment='top', horizontalalignment='right', 
                   bbox=box_props, fontweight='medium', family='monospace') # Monospace for alignment

    # ==========================================
    # --- Finalização e Salvamento ---
    # ==========================================
    # Agg backend handles plt.close() automatically, but we enforce it
    print(f"Salvando Dashboard de Features em: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight') 
    plt.close()

# ==========================================
# 1. CARREGAMENTO DOS MODELOS
# ==========================================
print('Carregando os modelos do disco...')
versao = '2'
model_a = tf.keras.models.load_model('nids_sensor_A_versao_{}.keras'.format(versao))
model_b = tf.keras.models.load_model('nids_sensor_B_versao_{}.keras'.format(versao))

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

# ==========================================
# 3. FUNÇÕES DE PLOTAGEM
# ==========================================
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
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true') # O TRUQUE DE NORMALIZAÇÃO!
    
    plt.figure(figsize=(20,16))
    sns.heatmap(cm_normalized, annot=cm_raw, fmt='d', annot_kws={"size": 8}, cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1)
    plt.title(title, fontsize=18)
    plt.ylabel('Classe verdadeira', fontsize=14)
    plt.xlabel('Classe predita', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. PREPARAÇÃO DOS DADOS
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
# 5. ATAQUE ADVERSARIAL E COLABORAÇÃO
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

epsilons = [0.002, 0.003]

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

    # ==========================================
    # TIER 2: ATRIBUIÇÃO MULTICLASSE (MAX CONFIDENCE)
    # ==========================================
    confs_A = np.max(predictions_adv_prob, axis=-1)
    system_final_preds_multi = np.where(confs_A > confs_B, predictions_adv, preds_B)

    print(f"\n--- MÉTRICAS DO SISTEMA CONJUNTO (MAX CONFIDENCE - Eps {eps_value}) ---")
    report_system = classification_report(y_test_malicious, system_final_preds_multi, 
                                          target_names=target_names, 
                                          labels=np.arange(num_classes), 
                                          zero_division=0)
    print(report_system)

    with open(f"relatorio_adv_SISTEMA_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(report_system)
        
    report_adv_A = classification_report(y_test_malicious, predictions_adv, target_names=target_names, labels=np.arange(num_classes), zero_division=0)
    report_adv_B = classification_report(y_test_malicious, preds_B, target_names=target_names, labels=np.arange(num_classes), zero_division=0)
    
    with open(f"relatorio_adv_A_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(report_adv_A)
    with open(f"relatorio_adv_B_eps_{eps_value}.txt", "w", encoding="utf-8") as f:
        f.write(report_adv_B)

    # PLOTAGEM MULTICLASSE (Tirando vantagem do Max Confidence)
    plot_confusion_matrix(y_test_malicious, predictions_adv, f'FGSM Multiclasse - Modelo A (Eps {eps_value})', f'matriz_fgsm_multi_A_eps_{eps_value}.png')
    plot_confusion_matrix(y_test_malicious, preds_B, f'FGSM Multiclasse - Modelo B (Eps {eps_value})', f'matriz_fgsm_multi_B_eps_{eps_value}.png')
    plot_confusion_matrix(y_test_malicious, system_final_preds_multi, f'FGSM Multiclasse - Sistema (Eps {eps_value})', f'matriz_fgsm_multi_sistema_eps_{eps_value}.png') # <-- O PLOT QUE FALTAVA!

    # ==========================================
    # TIER 1: DETECÇÃO BINÁRIA (OR-GATE)
    # ==========================================
    a_tricked = (predictions_adv == benign_index)
    b_tricked = (preds_B == benign_index)
    both_tricked = a_tricked & b_tricked
    
    total_a_evasions = np.sum(a_tricked)
    total_b_evasions = np.sum(b_tricked)
    total_system_evasions = np.sum(both_tricked)

    print("\n" + "-"*50)
    print("RESULTADOS DA DEFESA DISTRIBUÍDA (OR-GATE)")
    print("-"*50)
    print(f"Modelo A enganado sozinho: {total_a_evasions} fluxos")
    print(f"Modelo B enganado sozinho: {total_b_evasions} fluxos")
    print(f"-> Ambos enganados: {total_system_evasions} fluxos")

    system_evasion_rate = (total_system_evasions / len(X_test_malicious)) * 100
    print(f"Taxa de Evasão do Sistema Conjunto: {system_evasion_rate:.2f}%\n")

    total_attacks = len(X_test_malicious)
    rate_a_evasion = (total_a_evasions / total_attacks) * 100
    rate_b_evasion = (total_b_evasions / total_attacks) * 100

    texto_resumo = (
        f"RELATÓRIO DE EVASÃO DO SISTEMA (OR-GATE)\n"
        f"Epsilon da Carga (Ruído): {eps_value}\n"
        f"{'-'*50}\n"
        f"Total de fluxos maliciosos atacando o NIDS: {total_attacks}\n\n"
        f"Amostras que evadiram o Modelo A: {total_a_evasions} ({rate_a_evasion:.2f}%)\n"
        f"Amostras que evadiram o Modelo B: {total_b_evasions} ({rate_b_evasion:.2f}%)\n"
        f"Amostras que evadiram o Sistema Conjunto: {total_system_evasions} ({system_evasion_rate:.2f}%)\n"
    )
    
    nome_arquivo_resumo = f"resumo_evasao_sistema_eps_{eps_value}.txt"
    with open(nome_arquivo_resumo, "w", encoding="utf-8") as f:
        f.write(texto_resumo)

    # PLOTAGEM BINÁRIA
    true_adv_binary = np.ones(len(X_test_malicious), dtype=int)
    preds_A_binary = (~a_tricked).astype(int)
    preds_B_binary = (~b_tricked).astype(int)
    preds_system_binary = (~both_tricked).astype(int)
    
    plot_binary_confusion_matrix(true_adv_binary, preds_A_binary, f'FGSM Binário - Modelo A (Eps {eps_value})', f'matriz_fgsm_A_eps_{eps_value}.png')
    plot_binary_confusion_matrix(true_adv_binary, preds_B_binary, f'FGSM Binário - Modelo B (Eps {eps_value})', f'matriz_fgsm_B_eps_{eps_value}.png')
    plot_binary_confusion_matrix(true_adv_binary, preds_system_binary, f'FGSM Binário - Sistema (Eps {eps_value})', f'matriz_fgsm_sistema_eps_{eps_value}.png')

    # ==========================================
    # SALVAMENTO DO PICKLE
    # ==========================================
    if eps_value == 1.2:
        pickle_filename = f'dados_adversariais_eps_{eps_value}.pkl'
        print(f"Salvando flows adversariais em disco: {pickle_filename}")
        adversarial_dict = {
            'X_adv': X_test_adversarial,
            'y_true': y_test_malicious,
            'epsilon': eps_value
        }
        with open(pickle_filename, 'wb') as f:
            pickle.dump(adversarial_dict, f)
    
    l2_distance_mean = np.mean(np.linalg.norm(X_test_malicious - X_test_adversarial, axis=1))
    l_inf_distance_max = np.max(np.abs(X_test_malicious - X_test_adversarial))
    
    print(f"-> Magnitude do Ataque (L2 Média): {l2_distance_mean:.4f}")
    
    # 2. Gerar o Dashboard de Features (Passando as métricas)
    title_comp = f'Dashboard de Features Médias - FGSM (Eps {eps_value})'
    filename_comp = f'dashboard_features_eps_{eps_value}.png'
    
    plot_feature_comparison_dashboard(X_test_malicious, X_test_adversarial, title_comp, filename_comp, target_names, l2_distance_mean, l_inf_distance_max)
