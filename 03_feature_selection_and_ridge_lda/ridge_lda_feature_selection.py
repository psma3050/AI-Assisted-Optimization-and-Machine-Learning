import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
import time

data = load_breast_cancer()
X = data.data
y = data.target  #0 = benign, 1 = malignant
feature_names = list(map(str, data.feature_names))

#split data，確保每個fold的label有相同數量
def split_2fold_stratified(y):
    y = np.array(y)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    idx0 = np.sort(idx0)
    idx1 = np.sort(idx1)

    half0 = len(idx0) // 2
    half1 = len(idx1) // 2

    fold1 = np.concatenate([idx0[:half0], idx1[:half1]])
    fold2 = np.concatenate([idx0[half0:], idx1[half1:]])

    return fold1, fold2

# Balanced Accuracy
def balanced_accuracy(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    TPR = TP / (TP + FN )
    TNR = TN / (TN + FP )

    return (TPR + TNR) / 2

#LDA
def lda_predict_correct(X_train, y_train, X_test, C1=1.0, C2=1.0, ridge=0):
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError("lda_predict_correct requires exactly 2 classes in y_train")

    # compute per-class means
    X1 = X_train[y_train == 1]
    X2 = X_train[y_train == 0]
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    N = n1 + n2

    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)

    #calculate covariance matrix for each class
    S1 = np.atleast_2d(np.cov(X1.T)) if n1 > 1 else np.zeros((X.shape[1], X.shape[1]))
    S2 = np.atleast_2d(np.cov(X2.T)) if n2 > 1 else np.zeros((X.shape[1], X.shape[1]))

    #common covariance matrix
    Sigma = ((n1 - 1) * S1 + (n2 - 1) * S2) / (N - 2 + 1e-12)
    Sigma += np.eye(Sigma.shape[0]) * ridge
    Sigma_inv = np.linalg.inv(Sigma)

    #priors
    P1 = n1 / N
    P2 = n2 / N

    #weight vector and bias
    w = Sigma_inv @ (m1 - m2)
    b = -0.5 * (m1 - m2).T @ Sigma_inv @ (m1 + m2)
    b -= np.log((C1 * P2) / (C2 * P1) + 1e-12)

    preds = []
    for x in X_test:
        D = w.T @ x + b
        preds.append(1 if D >= 0 else 0)

    return np.array(preds)

#2fold-CV
def cross_val_2fold(X, y, classifier_func, use_stratified=True, **clf_kwargs):
    if use_stratified:
        fold1, fold2 = split_2fold_stratified(y)
    else:
        n = len(y)
        idx = np.arange(n)
        mid = n // 2
        fold1 = idx[:mid]
        fold2 = idx[mid:]

    pred1 = classifier_func(X[fold1], y[fold1], X[fold2], **clf_kwargs)
    acc1 = balanced_accuracy(y[fold2], pred1)

    pred2 = classifier_func(X[fold2], y[fold2], X[fold1], **clf_kwargs)
    acc2 = balanced_accuracy(y[fold1], pred2)

    return (acc1 + acc2) / 2

#SFS
def SFS(X, y, classifier_func=lda_predict_correct):
    N_features = X.shape[1]
    selected = []
    max_acc_record = []

    for k in range(N_features):
        best_f = None
        best_acc = -1
        #尚未被選過的特徵
        candidates = [f for f in range(N_features) if f not in selected]
        #依次加入候選特徵到selected計算CR
        for f in candidates:
            feature_set = selected + [f]
            acc = cross_val_2fold(X[:, feature_set], y, classifier_func, use_stratified=True)
            
            if acc > best_acc:
                best_acc = acc
                best_f = f
        #print(f"SFS Top-{k+1:2d}: Acc = {best_acc:.6f} | Features (names) = {[feature_names[i] for i in feature_set]}")
        selected.append(best_f)
        max_acc_record.append(best_acc)

        print(f"SFS Step {k+1:2d}: Select feature idx={best_f:2d} name='{feature_names[best_f]}', Acc = {best_acc:.4f}")

    best_k = int(np.argmax(max_acc_record))
    optimal_subset = selected[:best_k+1]

    return optimal_subset, selected, max_acc_record


#Fisher’s Criterion 
def fisher_score_all_features(X, y):
    scores = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        m0 = np.mean(xj[y == 0])
        m1 = np.mean(xj[y == 1])
        s0 = np.var(xj[y == 0])
        s1 = np.var(xj[y == 1])
        score = (m1 - m0)**2 / (s1 + s0 + 1e-12) #計算fisher score
        scores.append(score)
    return np.array(scores)

def fisher_feature_ranking(X, y):
    scores = fisher_score_all_features(X, y)
    ranking = np.argsort(scores)[::-1] 
    return ranking, scores

def fisher_evaluate_topN(X, y, classifier_func=lda_predict_correct):
    ranking, scores = fisher_feature_ranking(X, y)
    N_features = X.shape[1]
    acc_record = []
    best_acc = -1
    best_subset = None

    for N in range(1, N_features + 1):
        subset = ranking[:N]
        acc = cross_val_2fold(X[:, subset], y, classifier_func, use_stratified=True)
        acc_record.append(acc)
        new_f = ranking[N-1]
        print(f"Fisher Step {N:2d}: Add feature idx={new_f:2d} "
              f"name='{feature_names[new_f]}', Acc = {acc:.4f}")
        #print(f"Fisher Top-{N:2d}: Acc = {acc:.4f} | Features (names) = {[feature_names[i] for i in subset]}")
        if acc > best_acc:
            best_acc = acc
            best_subset = subset.copy()

    return best_subset, ranking, acc_record

def plot_CR_vs_subset(acc_record, title="CR vs Subset Number"):
    
    subset_numbers = np.arange(1, len(acc_record) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(subset_numbers, acc_record, marker='o')
    plt.xlabel("Subset Size (Number of Features)")
    plt.ylabel("Balanced Accuracy (CR)")
    plt.title(title)
    plt.grid(True)
    plt.xticks(subset_numbers) 
    plt.ylim(0.85,)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("\n\nPart 1: SFS")
    t0 = time.perf_counter()
    optimal_subset, selection_order, acc_record = SFS(X, y, classifier_func=lda_predict_correct)
    t1 = time.perf_counter()
    print("SFS optimal (size):", len(optimal_subset))
    print("SFS optimal subset (index, name):")
    for i in optimal_subset:
        print(f"  {i}: {feature_names[i]}")

    sfs_optimal_acc = cross_val_2fold(X[:, optimal_subset], y, lda_predict_correct)
    print(f"SFS optimal accuracy = {sfs_optimal_acc:.4f}")
    print(f"SFS 執行時間: {t1 - t0:.4f} 秒")

    print("\n\nPart 2: Fisher")
    t2 = time.perf_counter()
    fisher_best_subset, fisher_ranking, fisher_acc_record = fisher_evaluate_topN(X, y, classifier_func=lda_predict_correct)
    t3 = time.perf_counter()
      
    print("Fisher optimal (size):", len(fisher_best_subset))
    print("Fisher optimal subset (index, name):")
    for i in fisher_best_subset:
        print(f"  {i}: {feature_names[i]}")

    fisher_optimal_acc = cross_val_2fold(X[:, fisher_best_subset], y, lda_predict_correct)
    print(f"Fisher optimal accuracy = {fisher_optimal_acc:.4f}")
    print(f"Fisher 執行時間: {t3 - t2:.4f} 秒")
    #plot_CR_vs_subset(acc_record, title="SFS: CR vs Subset Number")
    #plot_CR_vs_subset(fisher_acc_record, title="Fisher: CR vs Subset Number")