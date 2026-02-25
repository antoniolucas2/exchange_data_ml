from mlp import *
import numpy as np

model_b = build_mlp()

print('PESOS DO MODELO QUE VAI RECEBER ANTES')

for w in model_b.get_weights():
    print(np.mean(w), np.std(w))

serve(model_b)

print('PESOS DO MODELO QUE RECEBEU DEPOIS')

for w in model_b.get_weights():
    print(np.mean(w), np.std(w))

