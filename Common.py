import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


cmap = ListedColormap(
    [[0.27, 0.67, 1.00], [1.00, 0.49, 0.46]]
)


input = np.array([
    [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0],
    [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
    [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
])

expected = np.array([
     0,  1,  1,
    -1,  0,  1,
    -1, -1,  0,
])


vals = np.linspace(-1.0, 1.0, 256)
test = np.array([[x, -y] for y in vals for x in vals])
