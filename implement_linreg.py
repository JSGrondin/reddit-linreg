import numpy as np
from numpy.linalg import inv

#Funtion for obtaining closed-Form Linear Regression solution
def cf_lin_reg(X, y):
    w = inv(X.T @ X) @ X.T @ y
    return w

###########################################
# Linear Regression using gradient descent
###########################################

# Check convergence using P-norm
def compute_Pnorm(w_0, w_1, p):
    tobesummed = np.power((w_1 - w_0), p)
    return np.sum(tobesummed) ** (1 / p)

# Function for calculating the mean squared error
def compute_mse(X, w, y):
    mse = ((X @ w - y) ** 2).mean(axis=None)
    return mse

def gd_lin_reg(X, y, eta_0, beta, eps):
    converged = False
    i = 1
    alpha = eta_0 / (1 + beta * i)
    w_0 = np.ones((len(X[0]), 1))  #array containing estimated w at each iteration i
    mat_X_transpose = np.transpose(X)
    mat_prod = mat_X_transpose @ X
    w_1 = w_0 - 2 * alpha * (mat_prod @ w_0 - mat_X_transpose @ y)

    #iterating to converge on w_arr
    while converged != True:
        w_0 = w_1
        w_1 = w_0 - 2 * alpha * (mat_prod @ w_0 - mat_X_transpose @ y)
        if compute_Pnorm(w_0, w_1, 2) <= eps:
            converged = True
        i += 1
        alpha = eta_0 / (1 + beta * i)
        print(i)
        print(compute_Pnorm(w_0, w_1, 2))
    return w_1

