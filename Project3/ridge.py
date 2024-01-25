import numpy as np


def transform(X):
        #insert a 1 at the beginning of each row
        ones=np.ones((X.shape[0],1))
        return np.concatenate((ones,X),axis=1)



class Ridge_Regression:
    def __init__(self, lambda_value=0, normalize=False):
        """
        Ridge regression model.
        
        Args:
            lambda_value = 0: regularization coefficient
            normalize = False: if True, normalize the data using Z-score normalization
        """
        self.lambda_value = lambda_value
        self.normalize = normalize
        self.weights = None

    def Train(self, X, y):
        """Train the ridge regression model using the closed form solution. The bias term is not regularized.
        
        Args:
            X: numpy array of dimension (N x M), where [N] is the number of training samples and [M] is the number of features
            y: numpy array of dimension (N x 1), where [N] is the number of training samples
        """
        
        if self.normalize:
            self.X_mean = np.mean(X,axis=0)
            self.X_std = np.std(X,axis=0)

            X = (X-self.X_mean)/self.X_std

        X = transform(X)
        I = np.eye(X.shape[1])
        I[0,0] = 0
        self.weights = np.linalg.inv(X.T.dot(X) + self.lambda_value * I).dot(X.T).dot(y)
        return

    def Predict(self, x):
        """
        Make a prediction on test samples x.
        Args:
            x: numpy array of test samples of dimension (N x M), where [N] is the number of test samples and [M] is the number of features
        Returns:
            numpy array of predictions of dimension (N x 1), where [N] is the number of test samples
        """
        if self.normalize:
            x = (x-self.X_mean)/self.X_std
        return transform(x).dot(self.weights)

    def __str__(self):
        return "Ridge Regression model with weights: ", self.weights
