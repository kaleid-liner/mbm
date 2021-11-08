from nn_meter import load_latency_predictor
from nn_meter.prediction.predictors.kernel_predictor import get_model

import numpy as np
import pandas as pd

import time
import pickle
import sys


device_map = {
    'cpu': 'cortexA76cpu_tflite21',
    'gpu': 'adreno640gpu_tflite21',
    'vpu': 'myriadvpu_openvino2019r2',
}
hardware = 'cpu'

predictor = load_latency_predictor(device_map[hardware])

results = []

for kernel, regressor in predictor.kernel_predictors.items():
    n_features = regressor.n_features_
    latency = 0
    for _ in range(0, 100):
        X = np.random.rand(1, n_features)
        start = time.time()
        y = regressor.predict(X)
        end = time.time()
        latency += end - start
    model_size = sys.getsizeof(pickle.dumps(regressor))
    results.append({
        'kernel': kernel,
        'latency (ms)': latency * 10,
        'size (byte)': model_size,
    })

df = pd.DataFrame(results)
df.to_csv('nn-meter_rf_latency.csv', index=False)
