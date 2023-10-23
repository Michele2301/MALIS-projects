
# main here

if __name__ == "__main__":
    import numpy as np
    from knn import KNN
    import pandas as pd
    # %%
    # read training.csv from the data folder
    dtr = pd.read_csv('../Data/training.csv')
    dtev = pd.read_csv('../Data/validation.csv')

    DTR = dtr.values
    DTEV = dtev.values
    accuracies = []

    for k in range(1, 100):
        knn_classifier = KNN(k)
        knn_classifier.train(DTR[:, :-1], DTR[:, -1])
        for p in range(1, 2):
            predictions = knn_classifier.predict(DTEV[:, :-1], p)
            accuracy = np.mean(predictions[:, 0] == DTEV[:, -1])
            print("k:", k, "p:", p, "accuracy:", accuracy)
            accuracies.append([k, p, accuracy])

    print(max(accuracies, key=lambda x: x[2]))

