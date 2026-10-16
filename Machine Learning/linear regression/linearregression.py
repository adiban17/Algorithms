import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(data):
    X = data[0]
    y = data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

