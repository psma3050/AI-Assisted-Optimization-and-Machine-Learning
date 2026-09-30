from qpsolvers import solve_qp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#load data
iris = load_iris()
X_all = iris.data[:, 2:4]      #use petal length & width only
y_all = iris.target

mask = (y_all == 1) | (y_all == 2)
X = X_all[mask]
y = y_all[mask]
y = np.where(y == 1, 1, -1) #select versicolor as pos (1), virginica as neg (-1)

#前25筆為training data, 後50筆為test data
X_train = np.vstack((X[0:25], X[50:75]))
y_train = np.hstack((y[0:25], y[50:75]))
X_test  = np.vstack((X[25:50], X[75:100]))
y_test  = np.hstack((y[25:50], y[75:100]))

np.set_printoptions(suppress=True)

def Linear_kernel(Xi, Xj):
    return np.dot(Xi.T, Xj)


#solve alpha
def Cal_Alpha(X_train, y_train, C):
    n = len(y_train)
    q = -np.ones(n)
    A = y_train.astype(float) #qpsolvers needs float
    b = np.array([0.0])
    lb = np.zeros(n) # 0 <= alpha_i
    ub = np.ones(n) * C # alpha_i <= C
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = y_train[i] * y_train[j] * Linear_kernel(X_train[i], X_train[j])
            #P = y_i y_j(x_i_T x_j)
    #Add a small identity matrix for numerical stability (regularization)
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


#compute bias using KT condition case 2
def KT_condition(alpha, X_train, y_train, C):
    b_star = 0
    #Find support vectors on the margin
    sv_idx = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]

    if len(sv_idx) > 0:
        #Average bias from all support vectors strictly on the margin
        b_sum = 0
        for k in sv_idx:
            temp = 0
            for i in range(len(y_train)): # Iterate over all training points
                if alpha[i] > 1e-6: # Only consider active alphas
                    temp += alpha[i] * y_train[i] * Linear_kernel(X_train[i], X_train[k])
            b_sum += (y_train[k] - temp)
        b_star = b_sum / len(sv_idx)
    else:
        # Fallback if no SVs are exactly on the margin (0<alpha, alpha=C)
        all_sv_indices = np.where(alpha > 1e-6)[0]
        if len(all_sv_indices) > 0:
            k = all_sv_indices[0]
            temp = 0
            for i in range(len(y_train)):
                if alpha[i] > 1e-6:
                    temp += alpha[i] * y_train[i] * Linear_kernel(X_train[i], X_train[k])
            b_star = y_train[k] - temp
        else:
            b_star = 0.0
            print("Warning: No support vectors found at all.")
            
    return np.round(b_star, 4)


#Test phase, find decision function
def Test_SVM(alpha, X_train, y_train, X_test, Y_test, b_star):
    count = 0
    for i in range(len(X_test)):
        temp = 0
        for j in range(len(alpha)):
            if alpha[j] > 1e-6: #Only consider active alphas
                temp += alpha[j] * y_train[j] * Linear_kernel(X_train[j], X_test[i])
        D = temp + b_star
        if (D >= 0 and Y_test[i] == 1) or (D < 0 and Y_test[i] == -1):
            count += 1   #D>0, pos; D<0, neg; D=0 margin
    CR = round(count / len(Y_test) * 100, 2)
    return CR

#calculate weight vector
def calculate_w(alpha, X_train, y_train):
    sv_indices = alpha > 1e-6
    alpha_y = alpha[sv_indices] * y_train[sv_indices]
    X_sv = X_train[sv_indices]
    w = np.dot(alpha_y, X_sv)
    return w

#plot the fighure
def plot_svm_boundary(w, b, X_train, y_train, alpha, C, CR, x_limits, y_limits):
    fig, ax = plt.subplots(figsize=(7, 6))
    #Plot data points
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='blue', label='Versicolor (+1)', alpha=0.7)
    ax.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], c='red', label='Virginica (-1)', alpha=0.7)

    #Find and circle SV
    sv_indices = alpha > 1e-6
    ax.scatter(X_train[sv_indices][:, 0], X_train[sv_indices][:, 1], 
               s=100, facecolors='none', edgecolors='k', 
               linewidth=1.5, label='Support Vectors')

    #Create grid to plot decision boundary
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    
    xx = np.linspace(x_min, x_max, 30)
    
    #Decision boundary: w[0]*x + w[1]*y + b = 0  =>  y = (-w[0]*x - b) / w[1]
    yy = (-w[0] * xx - b) / w[1]
    
    #Margin +1: w[0]*x + w[1]*y + b = 1   =>  y = (-w[0]*x - b + 1) / w[1]
    yy_plus1 = (-w[0] * xx - b + 1) / w[1]
    
    #Margin -1: w[0]*x + w[1]*y + b = -1  =>  y = (-w[0]*x - b - 1) / w[1]
    yy_minus1 = (-w[0] * xx - b - 1) / w[1]

    ax.plot(xx, yy, 'k-', label='Decision Boundary')
    ax.plot(xx, yy_plus1, 'k--', label='Margins') 
    ax.plot(xx, yy_minus1, 'k--')

    ax.set_title(f"Linear SVM (C = {C})\nTest Accuracy = {CR}%")
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.legend(loc='lower right', fontsize='small')
    ax.set_xlim(x_min, x_max) 
    ax.set_ylim(y_min, y_max) 
    
    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.show()

x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
global_x_limits = (x0_min, x0_max)
global_y_limits = (x1_min, x1_max)

print("Linear SVM:")
for C in [1.0, 10.0, 100.0]:
    print(f"C = {C}")
    alpha = Cal_Alpha(X_train, y_train, C)
    
    if alpha is not None:
        b_star = KT_condition(alpha, X_train, y_train, C)
        w_star = calculate_w(alpha, X_train, y_train)
        CR = Test_SVM(alpha, X_train, y_train, X_test, y_test, b_star)

        print(f"Alpha sum = {np.round(np.sum(alpha), 4)}")
        print(f"alpha = \n{np.round(alpha, 4)}")
        #print(f"w* = {np.round(w_star, 4)}")
        print(f"b* = {b_star}")
        print(f"CR = {CR}%\n")

        plot_svm_boundary(w_star, b_star, X_train, y_train, alpha, C, CR, global_x_limits, global_y_limits)
    