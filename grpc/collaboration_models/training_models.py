import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

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

training_csv = 'train/train.csv'
validation_csv = 'validation/validation.csv'
testing_csv = 'test/test.csv'

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

num_classes = len(label_encoder.classes_)
print("Total de labels:", num_classes)

print('Lendo o dataset de validação', validation_csv)

df_val = pd.read_csv(validation_csv)

for col in df_val.columns:
    if(col != 'label'):
        df_val[col] = pd.to_numeric(df_val[col], errors = 'coerce')

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

model_a = create_nids_mlp(input_dimension, num_classes=num_classes)
model_b = create_nids_mlp(input_dimension, num_classes=num_classes)

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

batch_size = 128
epochs = 20

early_stopper_a = EarlyStopping(monitor = 'val_loss', patience=4, restore_best_weights=True, verbose=1)
early_stopper_b = EarlyStopping(monitor = 'val_loss', patience=4, restore_best_weights=True, verbose=1)

print('Treinando o modelo A')

training_a = model_a.fit(X_train_A, y_train_A, validation_data=(X_val_A, y_val_A), batch_size=batch_size, epochs=epochs, callbacks=[early_stopper_a])

print('Treinando o modelo B')

training_b = model_b.fit(X_train_B, y_train_B, validation_data=(X_val_B, y_val_B), batch_size=batch_size, epochs=epochs, callbacks=[early_stopper_b])

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

predictions_A_prob = model_a.predict(X_test_A)
predictions_A = np.argmax(predictions_A_prob, axis=-1)

predictions_B_prob = model_b.predict(X_test_B)
predictions_B = np.argmax(predictions_B_prob, axis=-1)

target_names = label_encoder.classes_

print("Métricas de A:")
print(classification_report(y_test_A, predictions_A, target_names=target_names))

print("Métricas de B:")
print(classification_report(y_test_B, predictions_B, target_names=target_names))

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20,16))

    sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 8}, cmap='Blues', xticklabels=target_names, yticklabels=target_names)

    plt.title(title, fontsize=18)
    plt.ylabel('Classe verdadeira', fontsize=14)
    plt.xlabel('Classe predita', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test_A, predictions_A, 'Modelo A')
plot_confusion_matrix(y_test_B, predictions_B, 'Modelo B')

benign_index = list(label_encoder.classes_).index('BenignTraffic')

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

model_a.save("nids_sensor_A.keras")
model_b.save("nids_sensor_B.keras")
