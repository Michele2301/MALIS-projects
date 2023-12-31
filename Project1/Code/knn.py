"""
Project 1 - kNN
F. Dente, M. Ferrero
"""


import numpy
import numpy as np
import sys
from scipy.spatial import distance_matrix
import pandas as pd


class KNN:
    """
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    """

    def __init__(self, k):
        """
        INPUT :
        - k : is a natural number bigger than 0
        """

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")

        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k

    def train(self, X, y):
        """
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        """
        if X.shape[0] < self.k:
            raise Exception("Sorry, k is bigger than the number of training samples. Start again!")
        self.X = X
        # y are -1 o 1
        self.y = 2 * y - 1

    def predict(self, X_new, p, loop=True):
        """
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        """
        from scipy.stats import mode
        # compute the distance between X_new and X
        dst = self.minkowski_dist(X_new, p, loop)
        # Get the indices that would sort each row in ascending order
        nearest_neighbors = np.argpartition(dst, self.k, axis=1)[:, :self.k]
        # get the labels of the nearest neighbors, exploiting broadcasting
        nearest_neighbors_votes = self.y[nearest_neighbors]
        # get the most frequent label
        y = np.sum(nearest_neighbors_votes, axis=1)
        y_hat = np.where(y > 0, 1, 0)
        return y_hat.reshape((y_hat.size, 1))

    def minkowski_dist(self, X_new, p, loop=True):
        """
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance

        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        """
        dst = None
        if loop:
            # empty is faster than zeros
            dst = np.empty((X_new.shape[0], self.X.shape[0]))
            for i in range(X_new.shape[0]):
                # linalg is faster than raw minkowski distance
                dst[i, :] = np.linalg.norm(X_new[i, :] - self.X, ord=p, axis=1)
        else:
            dst = X_new[:, None, :] - self.X[None, :, :]
            # dst is now a 3D array of shape (M,N,D) thanks to broadcasting functionalities of numpy
            # now I can access a point using [x, y, z], where x indicates the point in X_new and y indicates the point in X with respect to which I am computing the distance and z the feature
            # now i need to sum over the last dimension (the features dimension) to obtain the distance between the two points
            dst = np.sum(np.abs(dst) ** p, axis=2) ** (1 / p)
        return dst


if __name__ == "__main__":
    dtr = pd.read_csv('../Data/training.csv')
    dtev = pd.read_csv('../Data/validation.csv')
    x_train = dtr.values[:, :-1]
    y_train = dtr.values[:, -1]
    x_val = dtev.values[:, :-1]
    y_val = dtev.values[:, -1]
    p = 5
    k = 23
    if len(sys.argv) == 3:
        k = int(sys.argv[1])
        p = int(sys.argv[2])
    kNN = KNN(k)
    kNN.train(x_train, y_train)
    y_pred = kNN.predict(x_val, p)
    print(np.mean(y_pred == y_val.reshape((y_val.size, 1))))