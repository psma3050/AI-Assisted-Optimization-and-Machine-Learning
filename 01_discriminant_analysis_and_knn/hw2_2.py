from qpsolvers import solve_qp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X_all = iris.data[:, 2:4]    
y_all = iris.target

mask = (y_all == 1) | (y_all == 2)
X = X_all[mask]
y = y_all[mask]
y = np.where(y == 1, 1, -1)

X_train = np.vstack((X[0:25], X[50:75]))
y_train = np.hstack((y[0:25], y[50:75]))
X_test  = np.vstack((X[25:50], X[75:100]))
y_test  = np.hstack((y[25:50], y[75:100]))

np.set_printoptions(suppress=True)

def RBF_kernel(Xi, Xj, sigma):  
    #Radial Basis Function (RBF) Kernel
    #K(xi, xj) = exp(-||xi - xj||^2 / (2 * sigma^2))
    norm_sq = np.linalg.norm(Xi - Xj)**2
    return np.exp(-norm_sq / (2 * sigma**2))

#calculate alpha
def Cal_Alpha_RBF(X_train, y_train, C, sigma):
    n = len(y_train)
    q = -np.ones(n)
    A = y_train.astype(float) # qpsolvers needs float
    b = np.array([0.0])
    lb = np.zeros(n)
    ub = np.ones(n) * C
    #P matrix = y_i y_j K(x_i, x_j)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = y_train[i] * y_train[j] * RBF_kernel(X_train[i], X_train[j], sigma)
    P = P + np.eye(n) * 1e-6

    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver='cvxopt')
    
    if alpha is None:
        print("Solver failed to find a solution.")
        return None

    eps = 1e-6 
    alpha[np.abs(alpha) < eps] = 0
    alpha[alpha > C - eps] = C
    alpha = np.round(alpha, 6)
    return alpha


#calculate bias using KT condition case 2
def KT_condition_RBF(alpha, X_train, y_train, C, sigma):
    b_star = 0
    sv_idx = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]

    if len(sv_idx) > 0:
        b_sum = 0
        for k in sv_idx:
            temp = 0
            for i in range(len(y_train)): 
                if alpha[i] > 1e-6:
                    temp += alpha[i] * y_train[i] * RBF_kernel(X_train[i], X_train[k], sigma)
            b_sum += (y_train[k] - temp)
        b_star = b_sum / len(sv_idx)
    else:
        all_sv_indices = np.where(alpha > 1e-6)[0]
        if len(all_sv_indices) > 0:
            k = all_sv_indices[0] 
            temp = 0
            for i in range(len(y_train)):
                if alpha[i] > 1e-6:
                    temp += alpha[i] * y_train[i] * RBF_kernel(X_train[i], X_train[k], sigma)
            b_star = y_train[k] - temp
        else:
            b_star = 0.0
            print("Warning: No support vectors found at all.")
            
    return np.round(b_star, 4)

#Test phase, find decision function
def Test_SVM_RBF(alpha, X_train, y_train, X_test, Y_test, b_star, sigma):
    count = 0
    for i in range(len(X_test)):
        temp = 0 
        for j in range(len(alpha)):
            if alpha[j] > 1e-6: 
                temp += alpha[j] * y_train[j] * RBF_kernel(X_train[j], X_test[i], sigma)
        D = temp + b_star # Decision function value
        if (D >= 0 and Y_test[i] == 1) or (D < 0 and Y_test[i] == -1):
            count += 1
    CR = round(count / len(Y_test) * 100, 2)
    return CR


def plot_svm_boundary_RBF(alpha, b, X_train, y_train, sigma, C, CR, x_limits, y_limits):
    fig, ax = plt.subplots(figsize=(7, 6)) 

    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='blue', label='Versicolor (+1)', alpha=0.7)
    ax.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], c='red', label='Virginica (-1)', alpha=0.7)

    sv_indices = np.where(alpha > 1e-6)[0]
    X_sv = X_train[sv_indices]
    y_sv = y_train[sv_indices]
    alpha_sv = alpha[sv_indices]
    
    ax.scatter(X_sv[:, 0], X_sv[:, 1], s=100, facecolors='none', edgecolors='k', linewidth=1.5, label='Support Vectors')

    x_min, x_max = x_limits
    y_min, y_max = y_limits
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x_grid = np.array([xx[i, j], yy[i, j]])
            f_x = 0.0
            for k in range(len(sv_indices)):
                f_x += alpha_sv[k] * y_sv[k] * RBF_kernel(X_sv[k], x_grid, sigma)
            Z[i, j] = f_x + b

    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

    ax.set_title(f"RBF SVM (C = {C}, sigma = {sigma})\nTest Accuracy = {CR}%")
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.legend(loc='lower right', fontsize='small')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


RBF_C = 10.0

x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
global_x_limits = (x0_min, x0_max)
global_y_limits = (x1_min, x1_max)

for sigma in [5.0, 1.0, 0.5, 0.1, 0.05]:
    print(f"C = {RBF_C}, sigma = {sigma}")
    alpha_rbf = Cal_Alpha_RBF(X_train, y_train, RBF_C, sigma)
    
    if alpha_rbf is not None:
        b_star_rbf = KT_condition_RBF(alpha_rbf, X_train, y_train, RBF_C, sigma)
        CR_rbf = Test_SVM_RBF(alpha_rbf, X_train, y_train, X_test, y_test, b_star_rbf, sigma)
        
        print(f"Alpha sum = {np.round(np.sum(alpha_rbf), 4)}")
        print(f"alpha = \n{np.round(alpha_rbf, 4)}")
        print(f"b* = {b_star_rbf}")
        print(f"CR = {CR_rbf}%\n")

        plot_svm_boundary_RBF(alpha_rbf, b_star_rbf, X_train, y_train, sigma, RBF_C, CR_rbf, global_x_limits, global_y_limits)
   