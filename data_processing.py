
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path='AAPL.csv'):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df1 = df.reset_index()['close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1_scaled = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    return df1_scaled, scaler

def split_data(data, train_ratio=0.65):
    training_size = int(len(data) * train_ratio)
    train_data = data[:training_size]
    test_data = data[training_size:]
    return train_data, test_data

def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
