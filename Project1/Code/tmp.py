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
    # for k in range(1, 30):
    #     knn_classifier = KNN(k)
    #     knn_classifier.train(DTR[:, :-1], DTR[:, -1])
    #     for p in range(1, 10):
    #         predictions = knn_classifier.predict(DTEV[:, :-1], p)
    #         knn = KNeighborsClassifier(n_neighbors=k, p=p)
    #         knn.fit(DTR[:, :-1], DTR[:, -1])
    #         sk_predictions = knn.predict(DTEV[:, :-1])
    #         accuracy_sk = np.mean(sk_predictions == DTEV[:, -1])
    #         accuracy = np.mean(predictions[:, 0] == DTEV[:, -1])
    #         print("k:", k, "p:", p, "accuracy:", accuracy)
    #         print("k:", k, "p:", p, "accuracy_sk:", accuracy_sk)
    #         accuracies.append([k, p, accuracy])
    #         accuracies_sk.append([k, p, accuracy_sk])
    #
    # print(max(accuracies, key=lambda x: x[2]))
    # print(max(accuracies_sk, key=lambda x: x[2]))
