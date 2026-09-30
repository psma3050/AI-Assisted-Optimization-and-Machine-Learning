import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = list(map(str, data.feature_names))

def split_2fold_stratified(y):
    y = np.array(y)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    half0 = len(idx0) // 2
    half1 = len(idx1) // 2
    fold1 = np.concatenate([idx0[:half0], idx1[:half1]])
    fold2 = np.concatenate([idx0[half0:], idx1[half1:]])
    return fold1, fold2

def balanced_accuracy(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    return (TPR + TNR) / 2

def lda_predict(X_train, y_train, X_test, ridge=1e-4):
    X1 = X_train[y_train == 1]
    X2 = X_train[y_train == 0]
    n1, n2 = X1.shape[0], X2.shape[0]
    N = n1 + n2
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)
    S1 = np.atleast_2d(np.cov(X1.T)) if n1 > 1 else np.zeros((X_train.shape[1], X_train.shape[1]))
    S2 = np.atleast_2d(np.cov(X2.T)) if n2 > 1 else np.zeros((X_train.shape[1], X_train.shape[1]))
    Sigma = ((n1 - 1) * S1 + (n2 - 1) * S2) / (N - 2 + 1e-12)
    Sigma += np.eye(Sigma.shape[0]) * ridge
    Sigma_inv = np.linalg.pinv(Sigma)
    w = Sigma_inv @ (m1 - m2)
    b = -0.5 * (m1 - m2).T @ Sigma_inv @ (m1 + m2) - np.log((n2) / (n1) + 1e-12)
    return np.array([1 if (w.T @ x + b) >= 0 else 0 for x in X_test])

def cross_val(X, y):
    f1, f2 = split_2fold_stratified(y)
    p1 = lda_predict(X[f1], y[f1], X[f2])
    p2 = lda_predict(X[f2], y[f2], X[f1])
    return (balanced_accuracy(y[f2], p1) + balanced_accuracy(y[f1], p2)) / 2

# SFS
n_feat = X.shape[1]
selected = []
sfs_accs = []
for k in range(n_feat):
    best_f, best_acc = None, -1
    for f in range(n_feat):
        if f not in selected:
            acc = cross_val(X[:, selected + [f]], y)
            if acc > best_acc:
                best_acc = acc
                best_f = f
    selected.append(best_f)
    sfs_accs.append(best_acc * 100)

# Fisher Score Ranking
scores = []
for j in range(n_feat):
    m0, m1 = np.mean(X[y == 0, j]), np.mean(X[y == 1, j])
    s0, s1 = np.var(X[y == 0, j]), np.var(X[y == 1, j])
    scores.append((m1 - m0)**2 / (s1 + s0 + 1e-12))
fisher_ranking = np.argsort(scores)[::-1]
fisher_accs = [cross_val(X[:, fisher_ranking[:n+1]], y) * 100 for n in range(n_feat)]

# Generate the plot
plt.figure(figsize=(9, 5.2), dpi=200)
x_axis = np.arange(1, n_feat + 1)
plt.plot(x_axis, sfs_accs, 'o-', color='#1f77b4', linewidth=2, label=f'Sequential Forward Selection (Best: {max(sfs_accs):.2f}%)')
plt.plot(x_axis, fisher_accs, 's--', color='#ff7f0e', linewidth=1.8, label=f'Fisher Criterion Ranking (Best: {max(fisher_accs):.2f}%)')
best_k = int(np.argmax(sfs_accs)) + 1
plt.scatter([best_k], [max(sfs_accs)], color='red', s=120, zorder=5, label=f'Optimal SFS Subset (k={best_k})')

plt.title('Feature Selection: Balanced Accuracy vs. Subset Size', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Selected Features', fontsize=11)
plt.ylabel('Balanced Accuracy (%)', fontsize=11)
plt.xticks(np.arange(1, n_feat + 1, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()

os.makedirs('docs/assets', exist_ok=True)
plt.savefig('docs/assets/feature_selection_results.png', dpi=200)
print(f"Generated feature_selection_results.png! Best SFS: {max(sfs_accs):.2f}% with {best_k} features.")
