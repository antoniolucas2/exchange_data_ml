import paho.mqtt.client as mqtt
import pickle

MODEL_ID = "B"

PUBLISH_TOPIC = f"models/{MODEL_ID}/weights"
SUBSCRIBE_TOPIC = f"models/{'B' if MODEL_ID == 'A' else 'A'}/weights"

def on_message(client, userdata, msg):
    weights = pickle.loads(msg.payload)
    model.set_weights(weights)
    print(f"✅ Received weights from peer")

client = mqtt.Client(client_id=f"model-{MODEL_ID}")
client.on_message = on_message
client.connect("broker.local", 1883)

client.subscribe(SUBSCRIBE_TOPIC)
client.loop_start()
