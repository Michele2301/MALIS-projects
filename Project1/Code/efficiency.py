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
    start = time()
    for k in range(1, 30):
        knn_classifier = KNN(k)
        knn_classifier.train(DTR[:, :-1], DTR[:, -1])
        for p in range(1, 10):
            predictions = knn_classifier.predict(DTEV[:, :-1], p)
            knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn.fit(DTR[:, :-1], DTR[:, -1])
            sk_predictions = knn.predict(DTEV[:, :-1])
            accuracy_sk = np.mean(sk_predictions == DTEV[:, -1])
            accuracy = np.mean(predictions[:, 0] == DTEV[:, -1])
            accuracies.append([k, p, accuracy])
            accuracies_sk.append([k, p, accuracy_sk])
    end = time()
    print("time:", end - start)
    print(max(accuracies, key=lambda x: x[2]))
    print(max(accuracies_sk, key=lambda x: x[2]))
