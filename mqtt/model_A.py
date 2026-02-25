import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
import pickle
import time

# --- Configuration ---
BROKER = "localhost"
PORT = 1883
TOPIC = "models/weights"

# 1. Build Model A
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model_a = create_model()

print("Training Model A with random data...")
X_train = np.random.random((100, 10))
y_train = np.random.randint(0, 2, (100, 1))
model_a.fit(X_train, y_train, epochs=5, verbose=0)

print('PESOS DO MODELO QUE VAI MANDAR')

for w in model_a.get_weights():
    print(np.mean(w), np.std(w))


weights = model_a.get_weights()
serialized_weights = pickle.dumps(weights)

print(f"Weights serialized. Payload size: {len(serialized_weights)} bytes.")

client = mqtt.Client(protocol=mqtt.MQTTv5)

try:
    client.connect(BROKER, PORT, 60)
    client.publish(TOPIC, serialized_weights, qos=1)
    print(f"Weights published to topic '{TOPIC}' successfully.")
except Exception as e:
    print(f"Failed to connect or publish: {e}")
finally:
    client.disconnect()
