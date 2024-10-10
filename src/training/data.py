import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def original_process(train, lags):
    attr = "Lane 1 Flow (Veh/15 Minutes)"
    direction_attr = "direction"

    # Read CSV file
    df1 = pd.read_csv(train, encoding="utf-8").fillna(0)

    # Normalize traffic flow
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # One-hot encode the 'direction' column for 8 possible directions
    encoder = OneHotEncoder(
        sparse_output=False, categories=[["N", "S", "E", "W", "NE", "NW", "SE", "SW"]]
    )
    direction_encoded = encoder.fit_transform(df1[direction_attr].values.reshape(-1, 1))

    # Debugging: Check direction encoding
    print("Direction encoded shape:", direction_encoded.shape)

    # Combine the flow and direction features
    features = np.hstack(
        [flow1.reshape(-1, 1), direction_encoded]
    )  # Now 1 (flow) + 8 (directions) = 9 features

    # Debugging: Check combined feature shape
    print("Features shape:", features.shape)

    train_data = []
    for i in range(lags, len(flow1)):
        train_data.append(features[i - lags : i + 1])

    train_data = np.array(train_data)
    np.random.shuffle(train_data)

    X_train = train_data[:, :-1]  # All features except the last one for training
    y_train = train_data[:, -1, 0]  # The target is the flow column

    # Debugging: Check X_train shape
    print("X_train shape before reshaping:", X_train.shape)

    print(f"Shape of y_train: {y_train.shape}")

    return X_train, y_train, scaler, encoder


def original_process_test(train, lags):
    attr = "Lane 1 Flow (Veh/15 Minutes)"
    direction_attr = "direction"
    # Read CSV file
    df1 = pd.read_csv(train, encoding="utf-8").fillna(0)
    # Normalize traffic flow
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    # One-hot encode the 'direction' column for 8 possible directions
    encoder = OneHotEncoder(
        sparse_output=False, categories=[["N", "S", "E", "W", "NE", "NW", "SE", "SW"]]
    )
    direction_encoded = encoder.fit_transform(df1[direction_attr].values.reshape(-1, 1))
    # Combine the flow and direction features
    features = np.hstack([flow1.reshape(-1, 1), direction_encoded])  # 9 features
    # Create lagged training data
    train_data = []
    for i in range(lags, len(flow1)):
        train_data.append(features[i - lags : i + 1])
    train_data = np.array(train_data)
    np.random.shuffle(train_data)
    X_data = train_data[:, :-1]  # All features except the last one for training
    y_data = train_data[:, -1, 0]  # The target is the flow column
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, shuffle=False
    )
    # Reshape X_train and X_test for LSTM/GRU (if necessary)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    # Debugging: Check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return X_train, X_test, y_train, y_test, scaler, encoder
