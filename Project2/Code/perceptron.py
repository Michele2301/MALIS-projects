"""
Project 2 - Perceptron
F. Dente, M. Ferrero
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Perceptron:
    def __init__(self):
        self.weights = np.empty(0)

    def train(self, X, y, Xval, yval, tries=10,alpha=0.001):
        """
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        - tries : is a natural number representing the number of tries to find the best perceptron hyperplane
        """
        # setting weights to 0
        self.weights = np.zeros(X.shape[1] + 1)
        # add the offset b
        X = PolynomialFeatures(1).fit_transform(X)
        Xval = PolynomialFeatures(1).fit_transform(Xval)
        # set the minimum to the loss function of some starting weights
        weights = np.random.rand(X.shape[1])
        min = loss_function(weights, Xval, yval)
        for i in range(tries):
            # generate random vector between 0 and 1 for each try
            while True:
                # we start with a random vector of weights, N+1 because of the offset b
                m = 0
                for i in range(X.shape[0]):
                    if (weights.transpose().dot(X[i]) * y[i]) <= 0:
                        weights = weights + alpha * X[i] * y[i]
                        m += 1
                if m == 0:
                    break
            # given what she said on the paper we should use the validation to find the best weights
            if loss_function(weights, Xval, yval) < min:
                min = loss_function(weights, Xval, yval)
                self.weights = weights
                # print("min: ", min)
            # change starting point
            # print("loss is: ", loss_function(weights, Xval, yval))
            weights = np.random.rand(X.shape[1])

    def predict(self, X):
        return np.sign(self.weights[1:].transpose().dot(X.transpose()) + self.weights[0])


def loss_function(weights, X, y):
    """
    INPUT :
    - weights : is a 1D numpy array containing the weights of the model (D+1 elements)
    - X : is a 2D Nx(D+1) numpy array containing the coordinates of points
    - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
    """
    value = -1 * ((weights.transpose() / abs(weights)).dot(X.transpose()) * y).sum()
    return value


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
    Xdata, Xtest, ydata, ytest = train_test_split(data.values[:, 0:-1], data.values[:, -1] * 2 - 1, test_size=0.2,
                                                  random_state=0, stratify=data.values[:, -1] * 2 - 1)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdata, ydata, test_size=0.2, random_state=0, stratify=ydata)
    perceptron = Perceptron()
    perceptron.train(Xtrain, ytrain, Xval, yval, tries=5000, alpha=0.0001)
    print(np.mean(perceptron.predict(Xval) == yval))
