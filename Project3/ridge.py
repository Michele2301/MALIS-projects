import numpy as np


def transform(X):
    result = np.zeros((X.shape[0], X.shape[1] + 1))
    for i in range(X.shape[0]):
        result[i] = np.append(X[i], 1)
    return result


class Ridge_Regression:
    def __init__(self, lambda_value=0):
        self.lambda_value = lambda_value
        self.weights = None

    def Train(self, X, y):
        X = transform(X)
        self.weights = np.linalg.inv(X.T.dot(X) + self.lambda_value * np.identity(X.shape[1])).dot(X.T).dot(y)
        return

    def Predict(self, x):
        return transform(x).dot(self.weights)

    def __str__(self):
        return "Ridge Regression model with weights: ", self.weights
