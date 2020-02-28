from .util.lina import nearestSPD

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
import scipy.linalg as la


### For dimension d>1, splits the parameter vector into d individual parameter vectors for each dimension

def beta_split(beta, d):
    if d == 1:
        return beta
    else:
        l_params = beta[1:]
        beta_d = np.concatenate((np.tile(np.array([[beta[0]]]), d).T, np.split(l_params, d)), axis=1)
        return beta_d


### Kernel class

class Kernel:

    def __init__(self, l_scale, stat_kern, int_kern, reg, int_2D):
        self.l_scale = l_scale
        self.stat_kern = stat_kern  # stationary kernel
        self.int_kern = int_kern
        self.reg = reg  # regularisation term of kernel
        self.kernel = self.kern
        self.name = "Kernel"
        self.D = 1  # This is automatically updated
        self.int_2D = int_2D

    def kern(self, x, y, beta):
        l_scaleX = self.l_scale(x, beta)
        l_scaleY = self.l_scale(y, beta)
        arg = np.abs(x - y) / np.sqrt(l_scaleX ** 2 + l_scaleY ** 2)
        return (np.exp(2 * beta[0]) * np.sqrt((l_scaleX * l_scaleY) / (l_scaleX ** 2 + l_scaleY ** 2)) * self.stat_kern(
            arg))

    def regD(self, X, beta, lambd):  # regularisation term for higher dimensions (built out of 1D reg term)
        if self.D == 1:
            return self.reg(X, beta, lambd)
        else:
            beta_d = beta_split(beta, self.D)
            X_T = X.T
            r = self.reg(X_T[0], beta_d[0], lambd)  # regularisation function is the same in each dimension
            for i in range(self.D - 1):
                r = r * self.reg(X_T[i + 1], beta_d[i + 1], lambd)
            return r


### Gaussian process class extends Kernel

class GaussianProcess:

    def __init__(self, Kernel, beta, X, Y, mesh=np.arange(0, 1.01, 0.01)):
        self.Kernel = Kernel  # Kernel object
        self.beta = beta
        self.X = X
        self.Y = Y
        self.D = len(X[0])  # dimension of Gaussian Process
        self.mesh = mesh
        self.Kernel.D = self.D
        self.kernel = self.Kernel.kernel
        self.cov_matrix = nearestSPD(self.cov_matrix_(self.mesh, self.mesh, self.beta))
        self.name = "GP"

    def cov_matrix_(self, X1, X2, beta):
        if self.D == 1:
            return self.kernel(X1.flatten()[:, np.newaxis], X2.flatten(), beta)
        else:
            betaD = beta_split(beta, self.D)
            X1_T, X2_T = X1.T, X2.T
            cov_mat = self.kernel(X1_T[0][:, np.newaxis], X2_T[0], betaD[0])
            for i in range(self.D - 1):
                cov_mat = cov_mat * self.kernel(X1_T[i + 1][:, np.newaxis], X2_T[i + 1], betaD[i + 1])
            return cov_mat

    def mean_(self, X):
        return np.zeros(len(X)) + np.mean(self.Y)

    def log_likelihood(self, X, Y, beta):
        return mvn.logpdf(Y, np.zeros(len(X)), nearestSPD(self.cov_matrix_(X, X, beta)), allow_singular=True)

    def fit(self, init, lambd, adapt=False):
        X, Y = self.X, self.Y

        loss = lambda beta: -self.log_likelihood(X, Y, beta) + self.Kernel.regD(X, beta, lambd)
        beta_fit = minimize(loss, self.beta, method='BFGS')

        print("beta: ", beta_fit.x, "\nMessage: ", beta_fit.message, "\nValue: ",
              beta_fit.fun, "\nSuccess: ", beta_fit.success, "\nLog-Likelihood: ",
              self.log_likelihood(self.X, self.Y, beta_fit.x))
        return beta_fit.x

    def sample(self, n_samps, X=None, Y=None, mesh=None):

        if X is None and Y is None:
            X, Y = self.X, self.Y
        if mesh is None:
            mesh = self.mesh
            K_mesh = self.cov_matrix
        else:
            K_mesh = self.cov_matrix_(mesh, mesh, self.beta)

        K_data = nearestSPD(self.cov_matrix_(X, X, self.beta))
        L_data = la.cholesky(K_data, lower=True)
        K_s = self.cov_matrix_(X, mesh, self.beta)
        L_solved = la.solve_triangular(L_data, K_s, check_finite=False, lower=True)
        post_mean_vec = self.mean_(mesh) + np.dot(L_solved.T, la.solve_triangular(L_data, Y - self.mean_(X).flatten(),
                                                                                  check_finite=False,
                                                                                  lower=True)).reshape((len(mesh),))
        L_post = la.cholesky(nearestSPD(K_mesh - np.dot(L_solved.T, L_solved)), lower=True)

        stdv = np.nan_to_num(np.sqrt(np.diag(K_mesh) - np.sum(L_solved ** 2, axis=0)))

        return (post_mean_vec.reshape(-1, 1) + np.dot(L_post, np.random.normal(size=(len(mesh), n_samps))),
                post_mean_vec.reshape(-1, 1), stdv)
