from math import sqrt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, CenteredNorm


input1 = np.array([
    [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0],
    [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
    [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
])

inputVals = np.linspace(-1.0, 1.0, 4)
input2 = np.array([[x, -y] for y in inputVals for x in inputVals])

expected1 = np.array([
    1, 1, 1,
    1, 1, 0,
    1, 0, 0,
]).reshape(-1, 1) # Convert from row vector to column vector

expected2 = np.array([
    0, 1, 1,
    1, 0, 0,
    1, 0, 0,
]).reshape(-1, 1)

expected3 = np.array([
    1, 1, 0, 0,
    0, 0, 1, 1,
    0, 1, 0, 1,
    1, 0, 1, 0,
]).reshape(-1, 1)

expected4 = np.array([
    1, 0, 0, 0,
    0, 1, 1, 1,
    0, 1, 0, 1,
    1, 0, 1, 0,
]).reshape(-1, 1)

# Matrix of data for testing
vals = np.linspace(-1.1, 1.1, 32)
test = np.array([[x, -y] for y in vals for x in vals])


# For each row pair in (row[i](a), row[i](b)), perform an outer product
def elemOuterF(a, b): return np.outer(a, b)
elemOuter = np.vectorize(elemOuterF, signature='(m),(n)->(m,n)')

def elemDotF(a, b): return a * b
elemDot = np.vectorize(elemDotF, signature='(m, n),(o)->(m, n)')


cmap1 = ListedColormap(
    [[0.28, 0.67, 1.00], [1.00, 0.49, 0.43]]
)
cmap2 = LinearSegmentedColormap.from_list(
    "",
    [[0.30, 0.73, 0.32], [0.28, 0.67, 1.00], [1.00, 0.49, 0.43], [0.30, 0.73, 0.32]]
)
cmap = cmap2
# Map colors so an output value of 0 corresponds to blue (cmap[1]) and 1
# corresponds to red (cmap[2]), with green (cmap[0], cmap[3]) idicating
# values < 0 or > 1
norm = norm=CenteredNorm(vcenter=0.5, halfrange=1.5)

def plotScatter(coords, data):
    plt.scatter(coords[:, 0], coords[:, 1], c=data, cmap=cmap, norm=norm)

def plot(input, data, expected):
    fig, ax = plt.subplots()

    nData = round(sqrt(len(data)))
    ax.imshow(
        data.reshape(nData, nData),
        cmap=cmap,
        extent=[-1.1, 1.1, -1.1, 1.1],
        norm=norm
    )
    ax.scatter(input[:, 0], input[:, 1], c=expected, cmap=cmap, norm=norm)

    plt.show()
