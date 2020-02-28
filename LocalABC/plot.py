from .kernels import gauss
from .util import lina

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.tri as mtri
import scipy.linalg as la
import numpy as np

mesh_plt = np.arange(0, 1.001, 0.001)


# Plots 1D posterior samples from a Gaussian process object with data from a given function
def plt_samples_1D(func, gp, n, alph, post=True):
    if post:
        X = gp.X.flatten()
        samples, mean, _ = gp.sample(n)
        cpy = np.copy(X)
        gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
        ax0 = plt.subplot(gs[0])
        ax0.plot(mesh_plt, func(mesh_plt), color='b')  # plot_mesh is defined globally
        ax0.plot(gp.mesh, mean, color='#C70039')
        ax0.plot(gp.mesh, samples, alpha=alph, color='r', linewidth=3)
        ax0.plot(X, gp.Y, 'bo')
        ax0.set_title(str(n) + " posterior samples of " + gp.name)
        ax1 = plt.subplot(gs[1])
        ax1.eventplot(cpy)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
    else:
        plt.plot(gp.mesh, np.random.multivariate_normal(np.zeros(len(gp.mesh)), gp.cov_matrix, n).T, alpha=alph,
                 color='r')


def plt_samples_2d(func, gp, points):
    X1 = gp.X.T[0]
    X2 = gp.X.T[1]
    plt.scatter(X1, X2, c="b")
    plt.scatter(points.T[0], points.T[1], c="r")


# Computes kernel smoothing for given data and smoothing kernel


def k_smoother(X, Y, kern, lambd):
    def smooth_func(x):
        return np.sum(Y * kern((x - X) / lambd)) / np.sum(kern((x - X) / lambd))
    return smooth_func


# Plots expected reduction in posterior integral variance for given 1D GP and input points


def plt_cost_1D(GP, E_vars, point_set):
    X = GP.X.flatten()
    smoother = k_smoother(point_set, E_vars, gauss, 1 / len(point_set))  # kern is defined globally
    smoothed_e_vars = np.zeros(len(mesh_plt))
    for i in range(len(mesh_plt)):
        smoothed_e_vars[i] = smoother(mesh_plt[i])
    cpy = np.copy(X)
    gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(point_set, E_vars, 'bo')
    for i in point_set:
        ax0.axvline(x=i)
    for i in X:
        ax0.axvline(x=i, color="red")
    ax0.plot(mesh_plt, smoothed_e_vars)  # plot_mesh is defined globally
    ax0.set_title("Expected reduction in variance of the integral of " + GP.name)
    ax1 = plt.subplot(gs[1])
    ax1.eventplot(cpy)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)


def plt_cost_2D(E_vars, point_set):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    X1, X2 = point_set.T[0], point_set.T[1]
    triang = mtri.Triangulation(X1, X2)
    ax.plot_trisurf(triang, E_vars, cmap="jet")


## Plots lengthscale function of a given 1D GP object


def plt_l_scale_1D(GP, rec=True):
    if rec:
        plt.plot(np.arange(0, 1.01, 0.01), 1 / GP.Kernel.l_scale(np.arange(0, 1.01, 0.01), GP.beta))
        plt.title("Inverse of lengthscale for " + GP.name)
    else:
        plt.plot(np.arange(0, 1.01, 0.01), GP.Kernel.l_scale(np.arange(0, 1.01, 0.01), GP.beta))
        plt.title("Lengthscale for " + GP.name)


## Plots errors of a 1D function:


def plt_error(func, GP_list, N):
    true_val = quad(func, 0, 1)[0]
    errors = []
    X = list(range(1, len(GP_list) + 1))
    for i in GP_list:
        # errors.append(np.abs(true_val - Int(i.mesh,i.sample(1)[1].T))[0]) #not a good method of computing integral of posterior mean
        errors.append(true_val)  # Update this
    plt.plot(X, errors)
    plt.title("Error of integral for " + GP_list[0].name)
    plt.xlim((0, N))


# Plots eigenvalues of covariance matrix:
def plt_eigvals(GP):
    eigs = la.eigvals(lina.nearestSPD(GP.cov_matrix))
    plt.plot(list(range(1, len(eigs) + 1)), np.flip(np.sort(np.log(eigs))))
    plt.title("Eigenvalue on mesh decay for " + GP.name)