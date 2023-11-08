# main here
from time import time

if __name__ == "__main__":
    import numpy as np
    from knn import KNN
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier

    # %%
    # read training.csv from the data folder
    dtr = pd.read_csv('../Data/training.csv')
    dtev = pd.read_csv('../Data/validation.csv')

    DTR = dtr.values
    DTEV = dtev.values
    accuracies = []
    accuracies_sk = []
    X = DTEV[:, :-1]
    y = DTEV[:, -1]

    # Create a grid of testing points
    h = .02  # space in the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx is the x-axis coordinate of the points in the test set
    # yy is the y-axis coordinate of the points in the test set
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # X_test contains the test set inputs (xx,yy)
    X_test = np.c_[xx.ravel(), yy.ravel()]
    print(X_test.shape)
    k = 27
    p = 2
    start = time()
    knn_classifier = KNN(k)
    knn_classifier.train(DTR[:, :-1], DTR[:, -1])
    predictions = knn_classifier.predict(X_test, p, loop=True)
    end = time()
    print("time:", end - start)