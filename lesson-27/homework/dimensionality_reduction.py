import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data
y=iris.target

target_names = iris.target_names

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
    
    def fit(self, X):
        X = np.array(X)

        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered)

        if self.n_components is None:
            self.n_components = X.shape[1]
        else:
            n_components = self.n_components


        self.components_ = Vt[:, :n_components]

    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_

        return X_centered @ self.components_

pca = PCA(n_components=3)
pca.fit(X)
print(pca.transform(X))