
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

    knn_classifier = KNN(4)
    knn_classifier.train(DTR[:, :-1], DTR[:, -1])
    predictions = knn_classifier.predict(DTEV[:, :-1], 2)
    accuracy = np.mean(predictions == DTEV[:, -1])
    print("accuracy:", accuracy)

