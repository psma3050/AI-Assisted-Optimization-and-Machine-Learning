import numpy as np
from qpsolvers import solve_qp
from sklearn.datasets import load_iris

np.set_printoptions(suppress=True)  

iris = load_iris()
data = iris.data
target = iris.target

#1st fold
X_train_1 = np.concatenate((data[0:25, :], data[50:75, :]), axis=0)       #c1 & c2
X_train_2 = np.concatenate((data[50:75, :], data[100:125, :]), axis=0)    #c2 & c3
X_train_3 = np.concatenate((data[0:25, :], data[100:125, :]), axis=0)     #c1 & c3
X_test_1 = np.concatenate((data[25:50, :], data[75:100, :], data[125:, :]), axis=0)

#2nd fold
X_train_4 = np.concatenate((data[25:50, :], data[75:100, :]), axis=0)    #c1 & c2
X_train_5 = np.concatenate((data[75:100, :], data[125:, :]), axis=0)     #c2 & c3
X_train_6 = np.concatenate((data[25:50, :], data[125:, :]), axis=0)      #c1 & c3
X_test_2 = np.concatenate((data[0:25, :], data[50:75, :], data[100:125, :]), axis=0)

#label
Y_train = np.ones(50)
Y_train[25:] = -1
Y_test = np.ones(75)
Y_test[25:50] = 2
Y_test[50:] = 3

#RBF kernel
def RBF_kernel(Xi, Xj, sigma):
    norm = np.linalg.norm(Xi - Xj) ** 2
    return np.exp(-norm / (2 * (sigma ** 2)))


def Cal_Alpha(X_train, Y_train, C, sigma):
    n = X_train.shape[0]
    q = -np.ones(n)
    A = Y_train.copy()
    b = np.array([0.0])
    lb = np.zeros(n)
    ub = C * np.ones(n)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = RBF_kernel(X_train[i], X_train[j], sigma)
    P = np.outer(Y_train, Y_train) * K

    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

    alpha = np.clip(alpha, 0, C) #把alpha限制在0跟C之間
    alpha = np.round(alpha, 6)
    return alpha

def KT_condition(alpha, X_train, Y_train, C, sigma):
    b_sum = 0
    count = 0
    for k in range(len(alpha)):
        if 0 < alpha[k] < C: #找最佳bias
            temp = 0
            for i in range(len(alpha)):
                if alpha[i] > 0:
                    temp += alpha[i] * Y_train[i] * RBF_kernel(X_train[i], X_train[k], sigma)
            b_sum += (1 / Y_train[k]) - temp
            count += 1
    return np.nan if count == 0 else b_sum / count

#根據決策函數判斷為1/-1，可以成功判斷代表同時回傳1，否則0(因bias不存在)
def SVM(X_train, Y_train, C, sigma, X_test):
    alpha = Cal_Alpha(X_train, Y_train, C, sigma)
    b = KT_condition(alpha, X_train, Y_train, C, sigma)
    if np.isnan(b):
        return 0, np.nan
    preds = []
    for x in X_test:
        f = sum(alpha[j] * Y_train[j] * RBF_kernel(x, X_train[j], sigma)
                for j in range(len(alpha)) if alpha[j] > 0)
        preds.append(1 if f + b >= 0 else -1) 
    return 1, np.array(preds)

def One_against_one(c12, c23, c13):
    out = np.zeros((len(c12), 3))
    out[:, 0] = np.where(c12 == -1, 2, 1)   #y=-1, class2; y=1, class1
    out[:, 1] = np.where(c23 == 1, 2, 3)    #y=1, class2; y=-1, class3
    out[:, 2] = np.where(c13 == -1, 3, 1)   #y=-1, class3; y=1, class1
    return out

def Voting(c12, c23, c13, Y_test):
    votes = One_against_one(c12, c23, c13)
    same, correct = 0, 0
    for i in range(votes.shape[0]):
        counts = [np.sum(votes[i] == 1),
                  np.sum(votes[i] == 2),
                  np.sum(votes[i] == 3)]
        if counts[0] == counts[1] == counts[2]:
            same += 1
        else:
            label = np.argmax(counts) + 1
            if label == Y_test[i]:
                correct += 1
    CR = correct / len(Y_test)
    return CR, same

C_values = [1, 5, 10, 50, 100, 500, 1000]
sigma_values = [1.05 ** k for k in range(-100, 101, 5)]

two_fold_CR = np.ones((len(sigma_values), len(C_values)))
first_CR = np.ones((len(sigma_values), len(C_values)))
second_CR = np.ones((len(sigma_values), len(C_values)))
same_voting_1 = np.ones((len(sigma_values), len(C_values)))
same_voting_2 = np.ones((len(sigma_values), len(C_values)))
bestCR = 0

for i, C in enumerate(C_values):
    for j, sigma in enumerate(sigma_values):
        # Fold 1
        e1, c1 = SVM(X_train_1, Y_train, C, sigma, X_test_1)
        e2, c2 = SVM(X_train_2, Y_train, C, sigma, X_test_1)
        e3, c3 = SVM(X_train_3, Y_train, C, sigma, X_test_1)
        # Fold 2
        e4, c4 = SVM(X_train_4, Y_train, C, sigma, X_test_2)
        e5, c5 = SVM(X_train_5, Y_train, C, sigma, X_test_2)
        e6, c6 = SVM(X_train_6, Y_train, C, sigma, X_test_2)

        if e1 * e2 * e3 * e4 * e5 * e6 == 0:
            print(f"No bias: ({C}, {round(sigma, 4)}) ")
            two_fold_CR[j, i] = np.nan
            first_CR[j, i] = np.nan
            second_CR[j, i] = np.nan
            same_voting_1[j, i] = np.nan
            same_voting_2[j, i] = np.nan
        else:
            CR1, same1 = Voting(c1, c2, c3, Y_test)
            CR2, same2 = Voting(c4, c5, c6, Y_test)
            avgCR = round(((CR1 + CR2) / 2) * 100, 2)
            CR1 = round(CR1 * 100, 2)
            CR2 = round(CR2 * 100, 2)
            two_fold_CR[j, i] = avgCR
            first_CR[j, i] = CR1
            second_CR[j, i] = CR2
            same_voting_1[j, i] = same1
            same_voting_2[j, i] = same2
            if avgCR > bestCR:
                bestCR = avgCR


print(two_fold_CR)
print("1st fold:\n",first_CR)
print("2nd fold:\n",second_CR)

for i, C in enumerate(C_values):
    for j, sigma in enumerate(sigma_values):
        if two_fold_CR[j, i] == bestCR:
            print(f"\nBest grid: [sigma, C] = [{round(sigma, 4)}, {C}], Best CR = {bestCR}%")

import pandas as pd
acc1_df = pd.DataFrame(
    first_CR,
    index=[f"σ={round(s,4)}" for s in sigma_values],
    columns=[f"C={c}" for c in C_values]
)
acc2_df = pd.DataFrame(
    second_CR,
    index=[f"σ={round(s,4)}" for s in sigma_values],
    columns=[f"C={c}" for c in C_values]
)

print(acc1_df.round(2))
print(acc2_df.round(2))

#acc_df = pd.DataFrame(
#    np.nan_to_num(two_fold_CR, nan=0.0),
#    index=[f"σ={round(s,4)}" for s in sigma_values],
#    columns=[f"C={c}" for c in C_values]
#)
#
#acc_df.to_excel('results1.xlsx')
