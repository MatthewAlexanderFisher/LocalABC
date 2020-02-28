import numpy as np
import scipy.linalg as la

### Calculates block cholesky decomposition


def block_chol(L,x):
    B = x[:-1]
    d = x[-1]
    tri = la.solve_triangular(L,B,check_finite = False, lower = True)
    return(np.block([
        [L, np.zeros((len(B),1))],
        [tri,np.sqrt(d - np.dot(tri,tri))]
    ]))


# Calculates nearest semi-positive definite matrix w.r.t. the Frobenius norm (algorithm based on Nick Higham's
# "Computing the nearest correlation matrix - a problem from finance"


def nearestSPD(A):
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))

    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    if la.norm(A - A3, ord='fro') / la.norm(A3, ord='fro') > 10:
        print("Matrix failed to be positive definite, distance in Frobenius norm: ",
              la.norm(A - A3, ord='fro') / la.norm(A3, ord='fro'))
    return A3


### Checks if input matrix is positive definite


def isPD(B):
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
