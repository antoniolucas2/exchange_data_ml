import tensorflow as tf
import paho.mqtt.client as mqtt
import pickle
import numpy as np

# --- Configuration ---
BROKER = "localhost"
PORT = 1883
TOPIC = "models/weights"

# 1. Build Model B (Must match Model A architecture)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model_b = create_model()

print('PESOS DO MODELO QUE VAI RECEBER ANTES')

for w in model_b.get_weights():
    print(np.mean(w), np.std(w))

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to broker with result code {rc}")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print(f"\nMessage received on {msg.topic}")
    
    try:
        received_weights = pickle.loads(msg.payload)
        
        model_b.set_weights(received_weights)
        print('PESOS DO MODELO QUE VAI RECEBER DEPOIS')

        for w in model_b.get_weights():
            print(np.mean(w), np.std(w))
       
        print("Successfully updated Model B weights from network.")
        
        client.disconnect()
        
    except Exception as e:
        print(f"Error applying weights: {e}")

client = mqtt.Client(protocol=mqtt.MQTTv5)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, PORT, 60)
    print("Waiting for weights...")
    client.loop_forever()
except KeyboardInterrupt:
    print("Stopping...")
    client.disconnect()
