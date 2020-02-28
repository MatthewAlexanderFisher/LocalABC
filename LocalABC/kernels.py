from . import gp

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm


### Length-scale functions

# length scale: linear interpolation of n points uniformly spread out
def l_linear_n(x, beta, n):
    cond_list = [np.logical_and(x >= i / n, x <= (i + 1) / n) for i in range(n)]
    choice_list = [n * (np.exp(beta[i + 1]) * ((i + 1) / n - x) + np.exp(beta[i + 2]) * (x - (i) / n)) for i in
                   range(n)]
    return np.select(cond_list, choice_list)


# length scale: piecewise constant of n points uniformly spread out
def l_pconst_n(x, beta, n):
    cond_list = [np.logical_and(x >= i / n, x <= (i + 1) / n) for i in range(n)]
    choice_list = [np.exp(beta[i + 1]) for i in range(n)]
    return np.select(cond_list, choice_list)


def l_explinear_n(x, beta, n):
    cond_list = [np.logical_and(x >= i / n, x <= (i + 1) / n) for i in range(n)]
    choice_list = [np.exp(n * (beta[i + 1] * ((i + 1) / n - x) + beta[i + 2] * (x - (i) / n))) for i in range(n)]
    return np.select(cond_list, choice_list)


# length scale: constant function
def l_const(x, beta):
    return x - x + np.exp(beta[1])


# length scale: Gaussian unknown location
def l_gauss_1D(x, beta):
    rv = norm(np.arctan(beta[2]) / np.pi + 0.5, np.exp(beta[3]))
    return 1 / (np.exp(beta[1]) + rv.pdf(x))


### Stationary kernels

# Matern smoothness 1/2
def matern_1(x):
    return np.exp(-x)


# Matern smoothness 3/2
def matern_2(x):
    return (1 + np.sqrt(3) * x) * np.exp(-np.sqrt(3) * x)


# Matern smoothness 5/2
def matern_3(x):
    return (1 + np.sqrt(5) * x + 5 / 3 * x ** 2) * np.exp(-np.sqrt(5) * x)


# Gaussian kernel
def gauss(x):
    return np.exp(-x ** 2)


### Regularisation functions

# regularisation function for piecewise linear length scale function
def r_lin_n(X, beta, lambd, n):  # lambd is the user input
    return (lambd[0] * 1 / (2 * n) * (np.sum(np.exp(beta[1:])) + np.sum(np.exp(beta[2:-1]))) + \
            lambd[1] * (1 / n) * np.sum((beta[2:] - beta[1:-1]) / (np.exp(beta[2:]) - np.exp(beta[1:-1]))))


# regularisation function for piecewise constant length scale function
def r_pconst_n(X, beta, lambd, n):
    return lambd[0] * np.mean(np.exp(beta[1:])) + lambd[1] * np.mean(np.exp(-beta[1:]))


# regularisation function for exponential of piecewise linear
def r_explinear_n(X, beta, lambd, n):
    _y = beta[2:] - beta[1:-1]
    return (lambd[0] * np.mean((np.exp(beta[2:]) - np.exp(beta[1:-1])) / _y) - \
            lambd[1] * np.mean((np.exp(-beta[2:]) - np.exp(-beta[1:-1])) / _y))


# regularisation function for constant length scale function
def r_const(X, beta, lambd):
    return lambd[0] * np.exp(beta[1]) + lambd[1] * np.exp(-beta[1])


# regularisation function for Gaussian 1D at unknown location
def r_gauss_1D(X, beta, lambd):
    rv = norm(np.arctan(beta[2]) / np.pi + 0.5, np.exp(beta[3]))
    return (lambd[0] * (rv.cdf(1) - rv.cdf(0) + np.exp(beta[1])) + lambd[1] / (
            rv.cdf(1) - rv.cdf(0) + np.exp(beta[1])) + lambd[2] * norm(-3, 2).pdf(np.exp(beta[3])))


### Integration functions

# 2D integration of kernel function (for the use of computing posterior variance)
def int_2D_n(fun, beta, n, simp):
    kern = lambda x, y: fun(x, y, beta)
    total = 0
    for i in range(n):
        for j in range(n):
            if simp is False:
                total = total + dblquad(kern, i / n, (i + 1) / n, j / n, (j + 1) / n)[0]
            else:
                total = total + dblquad(kern, i / n, (i + 1) / n, j / n, (j + 1) / n, epsabs=1e-5)[0]
    return total


# general integral for any kernel with any lengthscale
def int_lin_n(fun, x, beta, n, simp):
    kern = lambda y: fun(y, x, beta)
    total = 0
    for i in range(n):
        if simp is False:
            total = total + quad(kern, i / n, (i + 1) / n)[0]
        else:
            total = total + quad(kern, i / n, (i + 1) / n, epsabs=1e-5)[0]
    return total


# integral of Matern smoothness 1/2 with piecewise constant lengthscale
def int_pconst_mat1_n(fun, x, beta, n):
    c_x = l_pconst_n(x, beta, n)
    total = 0
    for i in range(n):
        c_y = np.exp(beta[i + 1])
        a = np.sqrt(c_y * c_x)
        b = np.sqrt(c_y ** 2 + c_x ** 2)
        exp1 = np.exp(-np.abs(x - (i + 1) / n) / b)
        exp2 = np.exp(-np.abs(x - i / n) / b)
        if x < i / n:
            total = total + a * (-exp1 + exp2)
        elif i / n <= x < (i + 1) / n:
            total = total + a * (2 - (exp1 + exp2))
        else:
            total = total + a * (exp1 - exp2)
    return np.exp(2 * beta[0]) * total


# integral of Matern smoothness 3/2 with piecewise constant lengthscale kernel
def int_pconst_mat2_n(fun, x, beta, n):
    c_x = l_pconst_n(x, beta, n)
    total = 0
    for i in range(n):
        c_y = np.exp(beta[i + 1])
        a = np.sqrt(c_y * c_x)
        b = np.sqrt(c_y ** 2 + c_x ** 2)
        exp1 = np.exp(-np.sqrt(3) * np.abs(x - (i + 1) / n) / b)
        exp2 = np.exp(-np.sqrt(3) * np.abs(x - i / n) / b)
        if x < i / n:
            lin1 = -3 * x + 2 * np.sqrt(3) * b + 3 * (i + 1) / n
            lin2 = -3 * x + 2 * np.sqrt(3) * b + 3 * i / n
            total = total + a / (3 * b) * (-exp1 * lin1 + exp2 * lin2)
        elif i / n <= x < (i + 1) / n:
            lin1 = -3 * x + 2 * np.sqrt(3) * b + 3 * (i + 1) / n
            lin2 = 3 * x + 2 * np.sqrt(3) * b - 3 * i / n
            total = total + 4 * a / np.sqrt(3) + a / (3 * b) * (- (lin1 * exp1 + lin2 * exp2))
        else:
            lin1 = 3 * x + 2 * np.sqrt(3) * b - 3 * (i + 1) / n
            lin2 = 3 * x + 2 * np.sqrt(3) * b - 3 * i / n
            total = total + a / (3 * b) * (exp1 * lin1 - exp2 * lin2)
    return np.exp(2 * beta[0]) * total


# integral of Matern smoothness 1/2 with constant lengthscale kernel
def int_const_mat1(fun, x, beta):
    c = np.exp(beta[1])
    return np.exp(2 * beta[0]) * c * (2 - np.exp(-x / c) - np.exp((x - 1) / c))


# integral of Matern smoothness 3/2 with constant lengthscale kernel
def int_const_mat2(fun, x, beta):
    c = np.exp(beta[1])
    b = np.sqrt(2) * c
    exp1 = np.exp(-np.sqrt(3) * (1 - x) / b)
    exp2 = np.exp(-np.sqrt(3) * x / b)
    lin1 = -3 * x + 2 * np.sqrt(3) * b + 3
    lin2 = 3 * x + 2 * np.sqrt(3) * b

    return np.exp(2 * beta[0]) * (4 * c / np.sqrt(3) + c / (3 * b) * (- (lin1 * exp1 + lin2 * exp2)))


# integral of constant lengthscale for kernel without closed-form integral
def int_const(fun, x, beta):
    kern = lambda y: fun(y, x, beta)
    return quad(kern, 0, 1)[0]


### Constructing kernel objects

# Linear lengthscale, Matern 1/2 kernel and 10 pieces
l_linear_10 = lambda x, beta: l_linear_n(x, beta, 10)
r_lin_10 = lambda X, beta, lambd: r_lin_n(X, beta, lambd, 10)
int_lin_10 = lambda fun, x, beta, simp=False: int_lin_n(fun, x, beta, 10, simp)
int_2D_10 = lambda fun, beta, simp=False: int_2D_n(fun, beta, 10, simp)

K_lin_mat1_10 = gp.Kernel(l_linear_10, matern_1, int_lin_10, r_lin_10, int_2D_10)

# linear lengthscale, Matern 3/2 kernel and 10 pieces
K_lin_mat2_10 = gp.Kernel(l_linear_10, matern_2, int_lin_10, r_lin_10, int_2D_10)

# linear lengthscale, Matern 3/2 kernel and 5 pieces
l_linear_5 = lambda x, beta: l_linear_n(x, beta, 5)
r_lin_5 = lambda X, beta, lambd: r_lin_n(X, beta, lambd, 5)
int_lin_5 = lambda fun, x, beta, simp=False: int_lin_n(fun, x, beta, 5, simp)
int_2D_5 = lambda fun, beta, simp=False: int_2D_n(fun, beta, 5, simp)
K_lin_mat2_5 = gp.Kernel(l_linear_5, matern_2, int_lin_5, r_lin_5, int_2D_5)

# linear lengthscale, Matern 3/2 kernel and 20 pieces
l_linear_20 = lambda x, beta: l_linear_n(x, beta, 20)
r_lin_20 = lambda X, beta, lambd: r_lin_n(X, beta, lambd, 20)
int_lin_20 = lambda fun, x, beta, simp=False: int_lin_n(fun, x, beta, 20, simp)
int_2D_20 = lambda fun, beta, simp=False: int_2D_n(fun, beta, 20, simp)
K_lin_mat2_20 = gp.Kernel(l_linear_20, matern_2, int_lin_20, r_lin_20, int_2D_20)

# linear lengthscale, Matern 5/2 kernel and 10 pieces
K_lin_mat3_10 = gp.Kernel(l_linear_10, matern_3, int_lin_10, r_lin_10, int_2D_10)

# piecewise constant lengthscale, Matern 1/2 and 10 pieces
l_pconst_10 = lambda x, beta: l_pconst_n(x, beta, 10)
r_pconst_10 = lambda X, beta, lambd: r_pconst_n(X, beta, lambd, 10)
int_pconst_mat1_10 = lambda fun, x, beta: int_pconst_mat1_n(fun, x, beta, 10)

K_pconst_mat1_10 = gp.Kernel(l_pconst_10, matern_1, int_pconst_mat1_10, r_pconst_10, int_2D_10)

# piecewise constant lengthscale, Matern 3/2 and 10 pieces
l_pconst_10 = lambda x, beta: l_pconst_n(x, beta, 10)
r_pconst_10 = lambda X, beta, lambd: r_pconst_n(X, beta, lambd, 10)
int_pconst_mat2_10 = lambda fun, x, beta: int_pconst_mat2_n(fun, x, beta, 10)

K_pconst_mat2_10 = gp.Kernel(l_pconst_10, matern_2, int_pconst_mat2_10, r_pconst_10, int_2D_10)

# exponential of piecewise linear lengthscale, Matern 3/2 kernel and 10 pieces
l_explinear_10 = lambda x, beta: l_explinear_n(x, beta, 10)
r_explinear_10 = lambda X, beta, lambd: r_explinear_n(X, beta, lambd, 10)

K_explin_mat2_10 = gp.Kernel(l_explinear_10, matern_2, int_lin_10, r_explinear_10, int_2D_10)

# constant length scale, Matern 1/2 kernel
K_const_mat1 = gp.Kernel(l_const, matern_1, int_const_mat1, r_const, int_2D_10)

# constant length scale, Matern 3/2 kernel
K_const_mat2 = gp.Kernel(l_const, matern_2, int_const_mat2, r_const, int_2D_10)

# Gaussian length scale at unknown location, Matern 3/2 kernel
K_gauss_mat2 = gp.Kernel(l_gauss_1D, matern_2, int_lin_10, r_gauss_1D, int_2D_10)
