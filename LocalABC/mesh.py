import numpy as np

### 1D get points function


def get_points(X):
    X = X.flatten()
    return (np.sort(X)[1:] + np.sort(X)[:-1]) / 2


# D>1 get points function - uses Voronoi cells
from scipy.spatial import Voronoi


def get_points_D(X, p):
    verts = Voronoi(X).vertices
    pointSet = []
    for i in verts:
        inCube = True
        for j in i:
            if j < 0 or j > 1:
                inCube = False
                break
        if inCube:
            pointSet.append(i)
    return np.array(pointSet)


# initial point sets for recursive points
points_2D = np.array(
    [[0, 0.25], [0, 0.5], [0, 0.75], [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1], [0.5, 0],
     [0.5, 0.25], [0.5, 0.75], [0.5, 1], [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1], [1, 0.25],
     [1, 0.5], [1, 0.75]])
mesh_2D = np.array(np.meshgrid(np.arange(0, 1.02, 0.02), np.arange(0, 1.02, 0.02))).reshape(2, -1).T


# add "bisected" points
def new_points_D(point_set, new_x):
    point_set = point_set[np.any(point_set != new_x, axis=1)]
    D = len(new_x)

    if D == 2:
        S = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])
        ll = np.array([0, 0])
        ur = np.array([1, 1])
    if D == 3:
        S = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])
        ll = np.array([0, 0, 0])
        ur = np.array([1, 1, 1])

    m = 0
    for i in new_x:
        n = len(str(i)) - 2
        if n > m:
            m = float(n)

    new_points = np.round(new_x + S * 2 ** (-m - 1), 10)

    inidx = np.all(np.logical_and(ll <= new_points, new_points <= ur), axis=1)
    new_points = new_points[inidx]

    return np.unique(np.concatenate((point_set, new_points)), axis=0)


# mesh search
def mesh_points_D(point_set, new_x):
    return point_set[np.any(point_set != new_x, axis=1)]


# mesh search 1D
def mesh_points_1(point_set, new_x):
    return point_set[point_set != new_x]
