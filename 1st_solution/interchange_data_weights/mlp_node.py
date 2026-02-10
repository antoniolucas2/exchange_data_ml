import sys
import time
import threading
import numpy as np
import tensorflow as tf
from concurrent import futures
import grpc

import mlp_service_pb2
import mlp_service_pb2_grpc

def numpy_to_proto(array):

    return mlp_service_pb2.NDArray(
        data=array.tobytes(),
        shape=array.shape,
        dtype=array.dtype.name
    )

def proto_to_numpy(proto_array):
    array = np.frombuffer(proto_array.data, dtype=proto_array.dtype)
    return array.reshape(proto_array.shape)

def create_mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

class MLPServiceServicer(mlp_service_pb2_grpc.MLPExchangeServicer):
    def __init__(self, node_id, model):
        self.node_id = node_id
        self.model = model

    def SendWeights(self, request, context):
        print(f"[{self.node_id}] Received WEIGHTS from {request.sender_id} (Step {request.step})")
        
        new_weights = [proto_to_numpy(w) for w in request.weights]

        print('Received weights from peer')

        for weight in new_weights:
            print(weight)
        
        current_weights = self.model.get_weights()
        averaged_weights = []
        for w_local, w_remote in zip(current_weights, new_weights):
            averaged_weights.append(0.5 * w_local + 0.5 * w_remote)
        
        self.model.set_weights(averaged_weights)
        print(f"[{self.node_id}] Weights merged successfully.")
        
        return mlp_service_pb2.Ack(success=True, message="Weights merged")

    def SendData(self, request, context):
        print(f"[{self.node_id}] Received DATA from {request.sender_id}")

        X = proto_to_numpy(request.features)
        y = proto_to_numpy(request.labels)
        
        print(f"[{self.node_id}] Training on received data...")
        self.model.fit(X, y, verbose=0, epochs=1)
        
        return mlp_service_pb2.Ack(success=True, message="Data trained")

class MLPNode:
    def __init__(self, node_id, port, peer_port):
        self.node_id = node_id
        self.port = port
        self.peer_port = peer_port
        self.model = create_mlp()
        self.stop_event = threading.Event()

    def start_server(self):

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        mlp_service_pb2_grpc.add_MLPExchangeServicer_to_server(
            MLPServiceServicer(self.node_id, self.model), server
        )
        server.add_insecure_port(f'[::]:{self.port}')
        server.start()
        print(f"[{self.node_id}] Server listening on port {self.port}")
        return server

    def run_client_loop(self):

        time.sleep(5)
        
        channel = grpc.insecure_channel(f'localhost:{self.peer_port}')
        stub = mlp_service_pb2_grpc.MLPExchangeStub(channel)
        
        step = 0
        while not self.stop_event.is_set():
            step += 1
            print(f"\n[{self.node_id}] --- Step {step} ---")
            
            X_train = np.random.rand(32, 10).astype(np.float32)
            y_train = np.random.randint(2, size=(32, 1)).astype(np.float32)
            
            print(f"[{self.node_id}] Training locally...")
            self.model.fit(X_train, y_train, verbose=0, epochs=1)

            print('Which action do you want choose to do?')
            action = input('0: Send weights to peer, 1: Send data to peer: ')
            
            if action == '0':
                print(f"[{self.node_id}] Sending WEIGHTS to peer...")

                print('Weights being sent:')

                for weight in self.model.get_weights():
                    print(weight)

                try:
                    weights_proto = [numpy_to_proto(w) for w in self.model.get_weights()]
                    stub.SendWeights(mlp_service_pb2.WeightUpdate(
                        sender_id=self.node_id,
                        weights=weights_proto,
                        step=step
                    ))
                except grpc.RpcError as e:
                    print(f"[{self.node_id}] Failed to send weights: {e.code()}")
            elif action == '1':
                print(f"[{self.node_id}] Sending DATA to peer...")
                try:
                    stub.SendData(mlp_service_pb2.DataBatch(
                        sender_id=self.node_id,
                        features=numpy_to_proto(X_train),
                        labels=numpy_to_proto(y_train)
                    ))
                except grpc.RpcError as e:
                    print(f"[{self.node_id}] Failed to send data: {e.code()}")
            
            time.sleep(3)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python mlp_node.py <node_id> <my_port> <peer_port>")
        sys.exit(1)
        
    node_id = sys.argv[1]
    my_port = sys.argv[2]
    peer_port = sys.argv[3]
    
    node = MLPNode(node_id, my_port, peer_port)
    server = node.start_server()
    
    try:
        node.run_client_loop()
    except KeyboardInterrupt:
        print("Stopping...")
        server.stop(0)
