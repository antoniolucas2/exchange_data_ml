import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

import pickle

import os

def oversample_dataset(X, y):
    print('Fazendo o oversample no dataset')

    class_counts = Counter(y)
    alvo = 2000

    over_strategy = {c: alvo for c, count in class_counts.items() if count < alvo}

    X_res, y_res = X, y

    if over_strategy:
        ros = RandomOverSampler(sampling_strategy=over_strategy, random_state=42)
        X_res, y_res = ros.fit_resample(X_res, y_res)
    
    print("Tamanho original:", len(y))
    print("Novo tamanho:", len(y_res))

    return X_res, y_res

def create_nids_mlp(input_dim, num_classes=34):

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

def print_binary_counts(y_true_bin, y_pred_bin, model_name):
    print("Resultado para", model_name)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    print("Trafego benigno\n")
    print("Classificado corretamente", tn)
    print("Falso alarme", fp)

    print("\nTrafego malicioso")
    print("Bloqueado corretamente", tp)
    print("Nao foi bloqueado", fn)

    print()

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

training_csv = 'train/train.csv'
validation_csv = 'validation/validation.csv'
testing_csv = 'test/test.csv'

print('Lendo o csv', training_csv)
df = pd.read_csv(training_csv)

print('Deixando as colunas apenas com numeros')
for col in df.columns:
    if col != 'label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X_train = df.drop(columns=['label'])
y_train = df['label']

del df

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print('Labels:', label_mapping)

numerical_features = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
one_hot_features = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

#preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', 'passthrough', one_hot_features)])

preprocessor = StandardScaler()

print('Padronizando as features')
X_train_processed = preprocessor.fit_transform(X_train)

del X_train
del y_train

print('Quantidade de fluxos dataset de treino:', len(X_train_processed))

X_train_processed, y_train_encoded = shuffle(X_train_processed, y_train_encoded, random_state=42)

num_classes = len(label_encoder.classes_)
print("Total de labels:", num_classes)

print('Lendo o dataset de validação', validation_csv)

df_val = pd.read_csv(validation_csv)

for col in df_val.columns:
    if(col != 'label'):
        df_val[col] = pd.to_numeric(df_val[col], errors = 'coerce')

df_val.replace([np.inf, -np.inf], np.nan, inplace=True)
df_val.dropna(inplace=True)

X_val = df_val.drop(columns=['label'])
y_val = df_val['label']

del df_val

X_val_processed = preprocessor.transform(X_val)
y_val_encoded = label_encoder.transform(y_val)

del X_val
del y_val

input_dimension = X_train_processed.shape[1]

print('Dimensao dos inputs:', input_dimension)

base_model = create_nids_mlp(input_dimension, num_classes=num_classes)
initial_weights = base_model.get_weights()

model_a = create_nids_mlp(input_dimension, num_classes=num_classes)
model_b = create_nids_mlp(input_dimension, num_classes=num_classes)

model_a.set_weights(initial_weights)
model_b.set_weights(initial_weights)

half_point_train = len(X_train_processed) // 2
half_point_validation = len(X_val_processed) // 2

X_train_A = X_train_processed[:half_point_train]
y_train_A = y_train_encoded[:half_point_train]

X_train_B = X_train_processed[half_point_train:]
y_train_B = y_train_encoded[half_point_train:]

X_val_A = X_val_processed[:half_point_validation]
y_val_A = y_val_encoded[:half_point_validation]

X_val_B = X_val_processed[half_point_validation:]
y_val_B = y_val_encoded[half_point_validation:]

#X_train_A, y_train_A = oversample_dataset(X_train_A, y_train_A)
#X_train_B, y_train_B = oversample_dataset(X_train_B, y_train_B)

batch_size = 256
epochs = 40

early_stopper_a = EarlyStopping(monitor = 'val_loss', patience=6, restore_best_weights=True, verbose=1)
early_stopper_b = EarlyStopping(monitor = 'val_loss', patience=6, restore_best_weights=True, verbose=1)

print('Treinando o modelo A')

training_a = model_a.fit(X_train_A, y_train_A, validation_data=(X_val_A, y_val_A), batch_size=batch_size, epochs=epochs, callbacks=[early_stopper_a])

print('Treinando o modelo B')

training_b = model_b.fit(X_train_B, y_train_B, validation_data=(X_val_B, y_val_B), batch_size=batch_size, epochs=epochs, callbacks=[early_stopper_b])

print('Lendo o dataset de teste')
df_test = pd.read_csv(testing_csv)

for col in df_test.columns:
    if col != 'label':
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
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

predictions_A_prob = model_a.predict(X_test_A)
predictions_A = np.argmax(predictions_A_prob, axis=-1)

predictions_B_prob = model_b.predict(X_test_B)
predictions_B = np.argmax(predictions_B_prob, axis=-1)

versao = 'mesmo_parente_diferente_dataset'
pasta_destino = "plots_e_resultados/treino_modelo/versao_{}".format(versao)
os.makedirs(pasta_destino, exist_ok=True)

nome_arquivo_matriz_imagem_A = "{}/matriz_modelo_A.png".format(pasta_destino) 
nome_arquivo_matriz_imagem_B = "{}/matriz_modelo_B.png".format(pasta_destino)
nome_arquivo_report_A = "{}/resultado_report_A.txt".format(pasta_destino)
nome_arquivo_report_B = "{}/resultado_report_B.txt".format(pasta_destino)
nome_arquivo_matriz_binaria_A = '{}/matriz_binaria_modelo_A.png'.format(pasta_destino)
nome_arquivo_matriz_binaria_B = '{}/matriz_binaria_modelo_B.png'.format(pasta_destino)
nome_modelo_A = 'nids_sensor_A_versao_{}.keras'.format(versao)
nome_modelo_B = 'nids_sensor_B_versao_{}.keras'.format(versao)
nome_arquivo_test_A = 'dados_originais_teste_A.pkl'
nome_arquivo_test_B = 'dados_original_teste_B.pkl'

# Salvando em formato pickle os dados originais

print('Salvando os dados de teste')
benign_index = list(label_encoder.classes_).index('BenignTraffic')
target_names = label_encoder.classes_

original_data_dict_A = {
        'X_test_A': X_test_A,
        'y_test_A': y_test_A,
        'target_names': target_names,
        'num_classes': len(target_names),
        'benign_index': benign_index
        }

with open(nome_arquivo_test_A, 'wb') as f_a:
    pickle.dump(original_data_dict_A, f_a)

original_data_dict_B={
        'X_test_B': X_test_B,
        'y_test_B': y_test_B,
        'target_names': target_names,
        'num_classes': len(target_names),
        'benign_index': benign_index
        }

with open(nome_arquivo_test_B, 'wb') as f_b:
    pickle.dump(original_data_dict_B, f_b)

print("Métricas de A:")
report_A_geral = classification_report(y_test_A, predictions_A, target_names=target_names)

print("Métricas de B:")
report_B_geral = classification_report(y_test_B, predictions_B, target_names=target_names)

with open(nome_arquivo_report_A, "w", encoding='utf-8') as f:
    f.write(report_A_geral)

with open(nome_arquivo_report_B, "w", encoding='utf-8') as f:
    f.write(report_B_geral)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20,16))

    sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 8}, cmap='Blues', xticklabels=target_names, yticklabels=target_names)

    plt.title(title, fontsize=18)
    plt.ylabel('Classe verdadeira', fontsize=14)
    plt.xlabel('Classe predita', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    print(f"Salvando matriz de confusão em: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight') 
    
    plt.close()

plot_confusion_matrix(y_test_A, predictions_A, 'Modelo A', nome_arquivo_matriz_imagem_A)
plot_confusion_matrix(y_test_B, predictions_B, 'Modelo B', nome_arquivo_matriz_imagem_B)

y_test_A_binary = (y_test_A != benign_index).astype(int)
predictions_A_binary = (predictions_A != benign_index).astype(int)

y_test_B_binary = (y_test_B != benign_index).astype(int)
predictions_B_binary = (predictions_B != benign_index).astype(int)

target_names_binary = ['Benign (0)', 'Malign (1)']

print('Metrica geral modelo A')
print(classification_report(y_test_A_binary, predictions_A_binary, target_names=target_names_binary))

print('Metrica geral modelo B')
print(classification_report(y_test_B_binary, predictions_B_binary, target_names=target_names_binary))

print_binary_counts(y_test_A_binary, predictions_A_binary, "Modelo A")
print_binary_counts(y_test_B_binary, predictions_B_binary, "Modelo B")

plot_binary_confusion_matrix(y_test_A_binary, predictions_A_binary, 'Modelo A (Binário)', nome_arquivo_matriz_binaria_A)
plot_binary_confusion_matrix(y_test_B_binary, predictions_B_binary, 'Modelo B (Binário)', nome_arquivo_matriz_binaria_B)

model_a.save(nome_modelo_A)
model_b.save(nome_modelo_B)

with open(nome_modelo_A.replace('.keras', '') + '_summary.txt', "w") as f:
    model_a.summary(print_fn=lambda x: f.write(x+"\n"))

with open(nome_modelo_B.replace('.keras', '') + '_summary.txt', 'w') as f:
    model_b.summary(print_fn=lambda x: f.write(x+"\n"))
