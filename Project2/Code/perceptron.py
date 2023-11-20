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
        alpha = 0.00001
        # add the offset b
        X = PolynomialFeatures(1).fit_transform(X)
        # generate random vector between 0 and 1
        for i in range(tries):
            weights = np.random.rand(X.shape[1])
            self.weights = weights
            min = (self.weights.reshape(1, self.weights.shape[0]).dot(X.transpose())).sum()
            while True:
                # we start with a random vector of weights, N+1 because of the offset b
                m = 0
                for i in range(X.shape[0]):
                    if (weights.transpose().dot(X[i]) * y[i]) <= 0:
                        weights = weights + alpha * X[i] * y[i]
                        m += 1
                if m == 0:
                    break
            if (weights.reshape(1,self.weights.shape[0]).dot(X.transpose())).sum()<min:
                min=(self.weights.reshape(1,self.weights.shape[0]).dot(X.transpose())).sum()
                self.weights = weights
            print((weights.reshape(1,self.weights.shape[0]).dot(X.transpose())).sum())

    def predict(self, X_new):
        print(self.weights.shape)
        return np.sign(self.weights[1:].reshape(1,self.weights[1:].shape[0]).dot(X_new.transpose()) + self.weights[0])

if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np

    data = load_digits(as_frame=True);
    data = data.frame
    data0 = data[data['target'] == 0]
    data1 = data[data['target'] == 1]
    data = pd.concat([data0, data1])
    Xdata, Xtest, ydata, ytest = train_test_split(data.values[:, 0:-1], data.values[:, -1]*2-1, test_size=0.2,
                                                  random_state=0, stratify=data.values[:, -1]*2-1)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdata, ydata, test_size=0.2, random_state=0, stratify=ydata)
    perceptron = Perceptron()
    perceptron.train(Xtrain, ytrain)
    print(np.mean(perceptron.predict(Xval) == yval))