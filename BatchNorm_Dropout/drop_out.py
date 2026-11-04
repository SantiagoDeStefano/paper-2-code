import numpy as np

weights = np.array([
    [24, 24.5, 24],
    [24, 24.5, 24],
    [24, 24.5, 24]
], dtype=float)

p = 0.5
mask = np.random.rand(*weights.shape)

for row in range(weights.shape[0]):
    for col in range(weights.shape[1]):
        if mask[row][col] > p:
            weights[row][col] = 0
        else:
            weights[row][col] /= (1 - p)

print(weights)