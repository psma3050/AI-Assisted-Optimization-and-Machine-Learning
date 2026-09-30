import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

#2-fold and split traning (first 25) and test (rest) data
def split_2fold(y):
    fold1_idx, fold2_idx = [], []
    for c in np.unique(y): # according to each class c, find all index
        idx = np.where(y == c)[0]
        fold1_idx.extend(idx[:25]) 
        fold2_idx.extend(idx[25:])
    return np.array(fold1_idx), np.array(fold2_idx)

#k-NN
def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test: #to each test data x
        dists = np.sqrt(np.sum((X_train - x) ** 2, axis=1)) #find the distance to all training data
        idx = np.argsort(dists)[:k] #find closest k data
        labels, counts = np.unique(y_train[idx], return_counts=True) #find their class and count times
        preds.append(labels[np.argmax(counts)]) #majority rule
    return np.array(preds)

#LDA
def lda_predict(X_train, y_train, X_test):
    classes = np.unique(y_train)
    means = []
    for c in classes:  #process each class
        subset = X_train[y_train == c]   #extract all data from the class
        mean_vec = np.mean(subset, axis=0)  #calculate their avg
        means.append(mean_vec)
    means = np.array(means)
    cov = np.cov(X_train.T) #common covariance matrix (from trainig data)
    cov_inv = np.linalg.inv(cov) #inverse
    preds = []
    for x in X_test:
        deci = []
        for i in range(len(classes)):
            mu = means[i] #mean vector of the i class
            deci_f = mu.T @ cov_inv @ x - 0.5 * mu.T @ cov_inv @ mu #decision function
            deci.append(deci_f)
        preds.append(classes[np.argmax(deci)])
    return np.array(preds)

#Cross validation
def cross_val_2fold(x, y, classifier_func):
    fold1, fold2 = split_2fold(y)

    #train=fold1,test=fold2
    pred1 = classifier_func(x[fold1], y[fold1], x[fold2]) #preidct result to test data from classifier by training feature, training label, test feature
    cr1 = np.mean(pred1 == y[fold2]) #use boolean array to calculate accuracy

    #train=fold2,test=fold1
    pred2 = classifier_func(x[fold2], y[fold2], x[fold1])
    cr2 = np.mean(pred2 == y[fold1])

    return (cr1 + cr2) / 2

#binary classification
feature_sets = {
    "All": [0, 1, 2, 3],   #all 4 features
    "PL+PW": [2, 3],       #PL+PW
    "SL+SW": [0, 1],       #SL+SW
}

pairs = [(0, 1), (0, 2), (1, 2)]

for fname, feat_idx in feature_sets.items():
    print(f"\nFeature set: {fname}")
    for (c1, c2) in pairs:
        mask = (y == c1) | (y == c2) #boolean mask, an array of T/F. True for any data point belonging to either c1 or c2.
        X_sub = x[mask][:, feat_idx] #select rows from x that correspond to the two classes of interest and columns specified by the current feature set
        y_sub = y[mask]

        acc_knn1 = cross_val_2fold(X_sub, y_sub, lambda a, b, c: knn_predict(a, b, c, k=1))
        acc_knn3 = cross_val_2fold(X_sub, y_sub, lambda a, b, c: knn_predict(a, b, c, k=3))
        acc_lda  = cross_val_2fold(X_sub, y_sub, lda_predict)

        print(f"C{c1+1} vs C{c2+1}: "f"1-NN={acc_knn1:.2f}, 3-NN={acc_knn3:.2f}, LDA={acc_lda:.2f}")
