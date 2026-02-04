from mlp import *
import numpy as np

model_a = build_mlp()

# Dummy training
x = np.random.rand(100, 10)
y = np.random.rand(100, 2)
model_a.fit(x, y, epochs=5, verbose=0)

print('PESOS DO MODELO QUE VAI MANDAR')

for w in model_a.get_weights():
    print(np.mean(w), np.std(w))


send_weights(model_a)

