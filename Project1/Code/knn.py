from scipy.spatial import distance_matrix
import numpy as np


class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")

        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k

    def train(self, X, y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''
        self.X = X
        self.y = y

    def predict(self, X_new, p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        '''
        ''' finds the indexes of the biggest k numbers'''
        indexes = np.argpartition(self.minkowski_dist(X_new, p), -self.k)[:, -self.k:]
        ''' map them to values, so now we have arrays of the nearest labels '''
        values = [[self.y[col] for col in col_indexes] for col_indexes in indexes]
        ''' still need to reduce to the most common'''
        ''' TODO '''
        return y_hat

    def minkowski_dist(self, X_new, p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        dst = []
        for i in range(0, X_new.size):
            for j in range(0, self.X.shape[0]):
                dst[i][j] = np.power(np.sum(np.power(np.abs(X_new[i] - self.X[j]), p)), 1 / p)
        return dst
