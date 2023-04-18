from math import sqrt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, CenteredNorm


input = np.array([
    [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0],
    [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
    [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
])

expected = np.array([
     0,  1,  1,
    -1,  0,  1,
    -1, -1,  0,
]).reshape(-1, 1) # Convert from row vector to column vector


# Matrix of data for testing
vals = np.linspace(-1.0, 1.0, 32)
test = np.array([[x, -y] for y in vals for x in vals])


# For each row pair in (row[i](a), row[i](b)), perform an outer product
def elemOuterF(a, b): return np.outer(a, b)
elemOuter = np.vectorize(elemOuterF, signature='(m),(n)->(m,n)')


cmap1 = ListedColormap(
    [[0.27, 0.67, 1.00], [1.00, 0.49, 0.46]]
)
cmap2 = LinearSegmentedColormap.from_list(
    "",
    [[0.27, 0.67, 1.00], [1.00, 0.49, 0.46]]
)
cmap = cmap2

def plotScatter(coords, data):
    plt.scatter(coords[:, 0], coords[:, 1], c=data, cmap=cmap)

def plot(data):
    fig, ax = plt.subplots()

    nData = round(sqrt(len(data)))
    ax.imshow(
        data.reshape(nData, nData),
        cmap=cmap,
        extent=[-1.0, 1.0, -1.0, 1.0],
        norm=CenteredNorm(halfrange=0.5)
    )
    plt.show()
