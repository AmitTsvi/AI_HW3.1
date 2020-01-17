import pandas as pd

def read_data(filename: str):
    data = pd.read_csv(filename, dtype=float)
    print(data.head())
    data = data.to_numpy()
    return data


def normalize_data(data, train_data):
    norm_data = (data - train_data.min(axis=0)) / train_data.ptp(axis=0)
    return norm_data