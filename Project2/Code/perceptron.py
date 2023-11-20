"""
Project 2 - Perceptron
F. Dente, M. Ferrero
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Perceptron:
    def __init__(self):
        self.weights = []

    def train(self, X, y, tries=10):
        # set the starting alpha
        alpha = 0.1
        # add the offset b
        X = PolynomialFeatures(1).fit_transform(X)
        while True:
            # we start with a random vector of weights, N+1 because of the offset b
            self.weights = np.random.randn(X.shape[0])
            m = 0
            for i in range(X.shape[0]):
                if ((self.weights.transpose().dot(X[i])) * y[i]) <= 0:
                    self.weights = self.weights + alpha * X[i] * y[i]
                    m += 1
            if m == 0:
                return

    def predict(self, X_new):
        return np.sign(self.weights[1:].transpose().dot(X_new) + self.weights[0])
