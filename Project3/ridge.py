import numpy as np


def transform(X):
        #insert a 1 at the beginning of each row
        ones=np.ones((X.shape[0],1))
        return np.concatenate((ones,X),axis=1)



class Ridge_Regression:
    def __init__(self, lambda_value=0):
        self.lambda_value = lambda_value
        self.weights = None

    def Train(self, X, y):
        X = transform(X)
        I = np.eye(X.shape[1])
        I[0,0] = 0
        self.weights = np.linalg.inv(X.T.dot(X) + self.lambda_value * I).dot(X.T).dot(y)
        return

    def Predict(self, x):
        return transform(x).dot(self.weights)

    def __str__(self):
        return "Ridge Regression model with weights: ", self.weights
