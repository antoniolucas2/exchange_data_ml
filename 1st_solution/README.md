# Solução usando gRPC

Fiz essa solução usando o gRPC. Ainda há muito o que ser estudado e melhorado nela, portanto quaisquer sugestões serão aceitas.

## Dependências

O programa depende do TensorFlow, do Keras e do gRPC para funcionar, além de Python. Para instalar, execute os seguintes comandos:

### TensorFlow
```sh
pip install tensorflow
```

### Keras
```sh
pip install keras
```

### gRPC
```sh
pip install grpcio
pip install grpcio-tools
```

## Como rodar

Primeiro, gere os protobufs:

```sh
sh generate_py_grpc.sh
```

Em seguida, sequencialmente, ative o server.py e o client.py

```sh
python server.py
python client.py
```

## Problemas até o momento

* O flow de dados não é bidirecional, cabendo ainda sua implementação.
* A conexão não é segura, pois não é encriptada.
