import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import MinMaxScaler

def get_dummy_data(n_features, n_clusters, n_samples=10000):
    csvname = "features-{}_clusters-{}".format(n_features, n_clusters)
    try:
        df = pd.read_csv("csvs/"+csvname)
        X1 = df.iloc[:, 1:n_features+1].values
        Y1 = df['y'].values
        print("Successfully read data from csv")
    except:
        print("csv not found, creating new csv")
        X1, Y1 = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters)
        scaler = MinMaxScaler()
        X1 = scaler.fit_transform(X1)
        df = pd.DataFrame(X1)
        df['y'] = Y1
        df.to_csv("csvs/"+csvname)