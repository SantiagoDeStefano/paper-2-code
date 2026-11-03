import numpy as np

weight_matrix = np.array([
    [-1273, 2, 100],
    [-5, 76, 52],
    [-123, 24, 9995]
], dtype=float)

epsilon = 1e-05

weight_mean = np.mean(weight_matrix, axis=0)
weight_var = np.var(weight_matrix, axis=0)

col_total = np.zeros(weight_matrix.shape[1])
col_norm = np.zeros(weight_matrix.shape[1])

for row in range(weight_matrix.shape[0]):
    for col in range(weight_matrix.shape[1]):
        weight_matrix[row][col] = ( weight_matrix[row][col] - weight_mean[col] ) / np.sqrt( weight_var[col] + epsilon )

print(weight_matrix)