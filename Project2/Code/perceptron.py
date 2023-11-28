"""
Project 2 - Perceptron
F. Dente, M. Ferrero
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def vcol(v):
        """
        Convert a 1D numpy array to a 2D column vector
        INPUT:
        - v : is a 1D numpy array
        OUTPUT:
        - a 2D numpy array with shape (v.size, 1)
        """
        return v.reshape((v.size, 1))

def vrow(v):
    """
    Convert a 1D numpy array to a 2D row vector
    INPUT:
    - v : is a 1D numpy array
    OUTPUT:
    - a 2D numpy array with shape (1, v.size)
    """
    return v.reshape((1, v.size))

class Perceptron:
    
    def __init__(self, alpha = 0.001):
        """
        Initialize the model
        INPUT :
        - alpha : is a real number representing the learning rate
        """
        self.alpha = alpha
        self.weights = np.empty(0)

    def __str__(self):
        return "Perceptron model with weights: ", self.weights

    def train(self, X, y, weights):
        """
        Train the model, 
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points where [N] is the number of samples and [D] is the number of features
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X where [N] is the number of samples and labels are -1 or 1
        - weights: is a 1D numpy array containing the initial weights of the model (D+1 elements)
        """
        # Add the offset b to the weights
        X = PolynomialFeatures(1).fit_transform(X)
        # Transform the weights to a column vector
        weights = vcol(weights)
        while True:
            m = 0
            for i in range(X.shape[0]):
                if (weights.transpose().dot(vcol(X[i])) * y[i]) <= 0:
                    weights = weights + self.alpha * vcol(X[i]) * y[i]
                    m += 1
            if m == 0:
                break    
        self.weights = weights

    def loss_function(self,X,y):
        return 0;

    def predict(self, X):
        return np.sign(self.weights[1:].transpose().dot(X.transpose()) + self.weights[0])



if __name__ == "__main__":

    data = load_digits(as_frame=True)
    data = data.frame
    data0 = data[data['target'] == 0]
    data1 = data[data['target'] == 1]
    data = pd.concat([data0, data1])
    Xdata, Xtest, ydata, ytest = train_test_split(data.values[:, 0:-1], data.values[:, -1] * 2 - 1, test_size=0.2,
                                                  random_state=0, stratify=data.values[:, -1] * 2 - 1)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdata, ydata, test_size=0.2, random_state=0, stratify=ydata)
    perceptron = Perceptron()
    perceptron.__init__(alpha=0.0001)
    perceptron.train(Xtrain, ytrain, np.zeros(Xtrain.shape[1] + 1))
    print(np.mean(perceptron.predict(Xval) == yval))
