from .gp import beta_split
from .plot import plt_l_scale_1D, plt_samples_1D, plt_cost_1D
from .util.save_out import create_out_dir, del_file, save_output
from .util.lina import block_chol, nearestSPD

import copy
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm


# Checks if optimisation of kernel parameters is necessary (only if new point is unexpected)
def get_optim(GP, new_x, new_y, chol, K_Xx, tol):
    solved = la.solve_triangular(chol, K_Xx, check_finite=False, lower=True)
    mu = np.dot(la.solve_triangular(chol, GP.Y, check_finite=False, lower=True),
                la.solve_triangular(chol, K_Xx, check_finite=False, lower=True))
    var = GP.kernel(new_x, new_x, GP.beta) - np.dot(solved, solved)
    z = (new_y - mu) / np.sqrt(var)

    if norm.cdf(np.abs(z)) > (1 + tol) / 2:
        return True
    else:
        return False


# Calculates integral parameters using mean and variance vector
def get_int_param(GP, mean_ints, chol):
    solved = la.solve_triangular(chol, mean_ints, check_finite=False, lower=True)

    mean_int = np.dot(la.solve_triangular(chol, GP.Y, check_finite=False, lower=True), solved)  # mean

    var_int = 1
    if GP.D == 1:
        var_int = GP.Kernel.int_2D(GP.kernel, GP.beta) - np.dot(solved, solved)  # variance
    else:
        beta_D = beta_split(GP.beta, GP.D)
        for i in range(GP.D):
            var_int = var_int * GP.Kernel.int_2D(GP.kernel, beta_D[i])
        var_int = var_int - np.dot(solved, solved)  # variance

    return np.array([mean_int, var_int])


# ABC algorithm in 1D
def ABC(func, _GP, get_points, n, lambd=[30, 1], point_mesh=False,
        options={"plot": True, "tol": False, "adapt": False, "save": False, "dir_name": "out"}):
    GP_list = []
    GP = copy.deepcopy(_GP)
    D = GP.D

    plot = options.get("plot")
    if plot is None:
        plot = True
    tol = options.get("tol")
    if tol is None:
        tol = False
    adapt = options.get("adapt")
    if adapt is None:
        adapt = False
    save = options.get("save")
    if save is None:
        save = False
    dir_name = options.get("dir_name")
    if dir_name is None:
        dir_name = "out"

    int_params = []  # list of integral estimates

    if tol == False:
        print("No threshold given, optimising beta at each step.")
    optim = True  # only if optim is true and we have a given tolerance we optimise the beta

    if point_mesh is not False:
        point_set = point_mesh

    if save is True:  # Create output directory if necessary
        create_out_dir(dir_name)

    for i in range(n + 1):  # The main loop
        print("Step ", i + 1, " of ", n + 1)
        if optim is True or tol is False:
            if i > 0 and tol != False:
                print("Optimising beta since new point outside tolerance ", z)
            GP.beta = GP.fit(GP.beta, lambd, adapt=adapt)
            GP.cov_matrix = GP.cov_matrix_(GP.mesh, GP.mesh, GP.beta)
        else:
            print("Not optimising beta since new point within tolerance.")

        GP_list.append(copy.deepcopy(GP))

        if D == 1 and plot == True:  # Plotting length scale
            plt_l_scale_1D(GP, True)
            plt.show()

            plt_samples_1D(func, GP, 40, 0.1)
            plt.show()

        if point_mesh is False:
            point_set = get_points(GP.X)

        kern1D_X_ints = np.zeros(len(GP.X) + 1)

        vars_i = []

        for j in range(len(GP.X)):
            kern1D_X_ints[j] = GP.Kernel.int_kern(GP.kernel, GP.X[j][0], GP.beta)  # this cannot be simplified otherwise our integral estimate would be incorrect

        mean_ints = kern1D_X_ints[:-1]
        kern = nearestSPD(GP.cov_matrix_(GP.X, GP.X, GP.beta))
        chol = la.cholesky(kern, lower=True)

        # mean and variance of integral
        solved = la.solve_triangular(chol, mean_ints, check_finite=False, lower=True)
        mean_int = np.dot(la.solve_triangular(chol, GP.Y, check_finite=False, lower=True), solved)
        var_int = GP.Kernel.int_2D(GP.kernel, GP.beta) - np.dot(solved, solved)
        int_params.append(np.array([mean_int, var_int]))

        if save is True:
            if i > 0:
                del_file(exp_name + ".pkl", dir_name)
            exp_name = GP.name + "_" + str(i)
            save_output([func, GP_list, np.array(int_params)], exp_name, dir_name)

        # if at last step then no need to find new point!
        if i == n:
            break

        for j in point_set:
            kern1D_X_ints[-1] = GP.Kernel.int_kern(GP.kernel, j,
                                                   GP.beta)  # This is the step we can reduce computation on integrals
            X_j = np.concatenate((GP.X, np.array([[j]])))
            K_Xj = GP.cov_matrix_(np.array([j]), X_j, GP.beta).flatten()
            chol_j = block_chol(chol, K_Xj)  # Computes cholesky decomp using cholesky decomp of previous kern matrix
            solved = la.solve_triangular(chol_j, kern1D_X_ints, check_finite=True,
                                         lower=True)  # This could break if points are not removed!
            var_j = -np.dot(solved, solved)

            vars_i.append(var_j)  # this is for plotting costs

        # Plot cost
        if D == 1 and plot == True:
            plt_cost_1D(GP, vars_i, point_set)
            plt.show()

        new_x = point_set[np.argmin(vars_i)]
        print(new_x)
        print(func(new_x))
        new_y = func(new_x)

        if not isinstance(new_y, float):  # checks if output is an integer or a numpy array with one element in
            new_y = new_y[0]

        if point_mesh is not False:
            point_set = get_points(point_set, new_x)
            print(point_set)

        # Check if optimisation is required:
        if tol != False:
            optim = get_optim(GP, new_x, new_y, chol, GP.cov_matrix_(np.array([new_x]), GP.X, GP.beta).flatten(), tol)

        # Update Gaussian process data:
        GP.X = np.concatenate((GP.X, np.array([[new_x]])))
        GP.Y = np.concatenate((GP.Y, np.array([new_y])))

    if plot == True:
        plt.plot(np.linspace(1, len(int_params), len(int_params)),
                 np.array(int_params).T[0])  # plot_integrals estimates
        plt.show()
    return GP_list, np.array(int_params)


# Performs ABC method in higher dimensions with assumed tensor product kernel
def ABC_D(func, _GP, get_points, n, lambd=[30, 1], point_mesh=False,
          options={"tol": False, "plot": False, "n_subset": "", "adapt": False, "save": False, "dir_name": "out"}):
    GP_list = []  # List of Gaussian processes used in output (a GP object for each i = 1,...,N)
    int_params = []  # list of posterior integral parameters (mean and variance) for each GP

    GP = copy.deepcopy(_GP)  # To ensure immutability of input GP
    D = GP.D

    plot = options.get("plot")
    if plot is None:
        plot = False
    tol = options.get("tol")
    if tol is None:
        tol = False
    n_subset = options.get("n_subset")
    if n_subset is None:
        n_subset = ""
    adapt = options.get("adapt")
    if adapt is None:
        adapt = False
    save = options.get("save")
    if save is None:
        save = False
    dir_name = options.get("dir_name")
    if dir_name is None:
        dir_name = "out"

    if tol is False:
        print("No threshold given, optimising beta at each step.")
    optim = True  # only if optim is true and we have a given tolerance we optimise the beta

    if point_mesh is not False:
        point_set = point_mesh

    if save is True:
        create_out_dir(dir_name)

    for i in range(n + 1):  # The main loop
        print("Step ", i + 1, " of ", n + 1)
        if optim == True or tol == False:
            if i > 0 and tol != False:
                print("Optimising beta since new point outside tolerance ", z)
            GP.beta = GP.fit(GP.beta, lambd, adapt=adapt)
            GP.cov_matrix = GP.cov_matrix_(GP.mesh, GP.mesh, GP.beta)
        else:
            print("Not optimising beta since new point within tolerance")

        GP_list.append(copy.deepcopy(GP))

        if plot is True:
            split = beta_split(GP.beta, D)
            for j in split:
                plt.plot(np.linspace(0, 1, 1000), 1 / GP.Kernel.l_scale(np.linspace(0, 1, 1000), j))
                plt.show()

        if point_mesh is False:
            point_set = get_points(GP.X, point_set)

        # Calculate all integrals at once:

        kernD_X_ints = np.zeros(len(GP.X) + 1) + 1
        point_set_ints = np.zeros(len(point_set)) + 1

        beta_D = beta_split(GP.beta, D)
        X_T = GP.X.T
        p_T = point_set.T
        for j in range(D):
            X_j = X_T[j]
            intsX_j = np.concatenate((X_j, np.array([0])))
            for k in np.unique(X_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])
                intsX_j = np.where(intsX_j == k, int_k, intsX_j)
            kernD_X_ints = kernD_X_ints * intsX_j

            p_j = p_T[j]
            intsp_j = p_j
            for k in np.unique(p_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])  # could potentially make this cheaper
                intsp_j = np.where(intsp_j == k, int_k, intsp_j)
            point_set_ints = point_set_ints * intsp_j

        chol = la.cholesky(nearestSPD(GP.cov_matrix_(GP.X, GP.X, GP.beta)), lower=True)
        int_param = get_int_param(GP, kernD_X_ints[:-1], chol)
        int_params.append(int_param)  # Add integral parameters to int_params
        print("(mean,var): ", int_param)

        if save is True:
            if i > 0:
                del_file(exp_name + ".pkl", dir_name)
            exp_name = GP.name + "_" + str(i)
            save_output([func, GP_list, np.array(int_params)], exp_name, dir_name)

        # if at last step then no need to find new point!
        if i == n:
            break

        # Compute optimisation on point set:
        vars_i = []
        if isinstance(n_subset, int):
            subset = np.random.choice(len(point_set), n_subset, replace=False)
            for j in subset:
                kernD_X_ints[-1] = point_set_ints[j]
                X_j = np.concatenate((GP.X, np.array([point_set[j]])))

                K_Xj = GP.cov_matrix_(np.array([point_set[j]]), X_j, GP.beta).flatten()
                chol_j = block_chol(chol, K_Xj)
                solved = la.solve_triangular(chol_j, kernD_X_ints, check_finite=False, lower=True)
                var_j = -np.dot(solved, solved)

                vars_i.append(var_j)
            n_subset = n_subset - 1
            new_x = point_set[subset[np.argmin(vars_i)]]
            new_y = func(new_x)
        else:
            for j in range(len(point_set)):
                kernD_X_ints[-1] = point_set_ints[j]
                X_j = np.concatenate((GP.X, np.array([point_set[j]])))

                K_Xj = GP.cov_matrix_(np.array([point_set[j]]), X_j, GP.beta).flatten()
                chol_j = block_chol(chol, K_Xj)
                solved = la.solve_triangular(chol_j, kernD_X_ints, check_finite=False, lower=True)
                var_j = -np.dot(solved, solved)

                vars_i.append(var_j)
            new_x = point_set[np.argmin(vars_i)]
            new_y = func(new_x)

        # Check if optimisation is required:
        if tol != False:
            optim = get_optim(GP, new_x, new_y, chol, GP.cov_matrix_(np.array([new_x]), GP.X, GP.beta).flatten(), tol)

        if point_mesh is not False:
            point_set = get_points(point_set, new_x)

        # Add new point to Gaussian process:
        GP.X = np.concatenate((GP.X, np.array([new_x])))
        GP.Y = np.concatenate((GP.Y, np.array([new_y])))

        if GP.D == 2 and plot is True:
            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(111)
            ax.scatter(GP.X.T[0], GP.X.T[1])
            plt.show()
        if GP.D == 3 and plot is True:
            fig = plt.figure(figsize=(16, 16))
            ax = Axes3D(fig)
            ax.scatter(GP.X.T[0], GP.X.T[1], GP.X.T[2])
            plt.show()
    return GP_list, np.array(int_params)
