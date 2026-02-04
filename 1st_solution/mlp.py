import tensorflow as tf
from keras import layers, models
import threading, time

def build_mlp():
    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(10,)),
        layers.Dense(2)
    ])
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model

def serialize_weights(model):
    tensors = []
    for w in model.get_weights():
        tensors.append({
            "values": w.flatten().tolist(),
            "shape": list(w.shape)
        })
    return tensors

import numpy as np

def load_weights(model, tensors):
    weights = []
    for tensor in tensors:
        arr = np.array(tensor.values, dtype=np.float32)
        weights.append(arr.reshape(tensor.shape))
    model.set_weights(weights)

import grpc
import model_pb2, model_pb2_grpc

def send_weights(model):
    channel = grpc.insecure_channel("localhost:50051")
    stub = model_pb2_grpc.ModelExchangeStub(channel)

    tensors = serialize_weights(model)

    proto_tensors = [
        model_pb2.Tensor(values=t["values"], shape=t["shape"])
        for t in tensors
    ]

    response = stub.SendWeights(
        model_pb2.ModelWeights(tensors=proto_tensors)
    )
    print(response.msg)

import grpc
from concurrent import futures
import model_pb2, model_pb2_grpc

class ModelServer(model_pb2_grpc.ModelExchangeServicer):
    def __init__(self, model, server):
        self.model = model
        self.server=server

    def SendWeights(self, request, context):
        load_weights(self.model, request.tensors)
        print("✅ Weights received and loaded into Model B")

        context.add_callback(lambda: self.server.stop(grace=1))

        return model_pb2.Ack(msg="Weights applied")

def serve(model):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = ModelServer(model, server)
    model_pb2_grpc.add_ModelExchangeServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("🚀 TF gRPC server running on port 50051")
    server.wait_for_termination()

