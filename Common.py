import numpy as np

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "",
    [[0.10, 0.58, 0.82], [0.50, 0.55, 0.62], [0.89, 0.39, 0.32]]
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


vals = np.linspace(-1.0, 1.0, 11)
test = np.array([[x, -y] for y in vals for x in vals])
