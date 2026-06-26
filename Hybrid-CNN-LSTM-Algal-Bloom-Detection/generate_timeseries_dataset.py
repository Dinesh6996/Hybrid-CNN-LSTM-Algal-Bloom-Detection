import numpy as np
import pandas as pd

np.random.seed(42)

samples = 20000
timesteps = 5

X, y = [], []

for _ in range(samples):
    seq = []
    bloom_flag = 0

    base_temp = np.random.uniform(20,30)
    base_nitrate = np.random.uniform(0.1,2)

    for t in range(timesteps):

        temperature = base_temp + np.random.uniform(-1,1)
        ph = np.random.uniform(6.5,8.5)
        do = np.random.uniform(3,10) - (t * 0.5)
        turbidity = np.random.uniform(1,20)
        nitrate = base_nitrate + (t * np.random.uniform(0.2,0.6))
        phosphate = np.random.uniform(0.1,2)
        chlorophyll = np.random.uniform(5,50) + (t * 5)

        # Improved bloom logic
        if (do < 3) or (nitrate > 2.5 and chlorophyll > 30):
            bloom_flag = 1

        seq.append([temperature, ph, do, turbidity, nitrate, phosphate, chlorophyll])

    X.append(seq)
    y.append(bloom_flag)

X = np.array(X)
y = np.array(y)

np.save("X_timeseries.npy", X)
np.save("y_timeseries.npy", y)

print("Dataset generated and saved!")