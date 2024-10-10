import sys

sys.dont_write_bytecode = True

from tcn import TCN
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import training.data as data
from train import MODELS, TEST_CSV_DIRECTION

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

MODEL_DIR = "./saved_models"
CSV_DIR = "../training_data/new_traffic_flows"


def plot_results(y_true, y_pred):
    d = "2016-10-1 00:00"
    x = pd.date_range(d, periods=96, freq="15min")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label="True Data")
    ax.plot(x, y_pred, label="Model")

    plt.legend()
    plt.grid(True)
    plt.xlabel("Time of Day")
    plt.ylabel("Flow")

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def predict_traffic_flow(time_input, direction_input, model_path, data_path):
    # print current directory
    print("Current Directory -> ", os.getcwd())

    # Load the model
    if "tcn" in model_path.lower():
        model = load_model(model_path, custom_objects={"TCN": TCN})
    else:
        model = load_model(model_path)

    # Load and preprocess the data
    df = pd.read_csv(data_path, encoding="utf-8").fillna(0)
    attr = "Lane 1 Flow (Veh/15 Minutes)"
    direction_attr = "direction"  # Direction column

    # Normalize traffic flow
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
    flow = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # One-hot encode the entire direction column for 8 possible directions
    encoder = OneHotEncoder(
        sparse_output=False, categories=[["N", "S", "E", "W", "NE", "NW", "SE", "SW"]]
    )
    direction_encoded = encoder.fit_transform(df[direction_attr].values.reshape(-1, 1))

    # Combine flow and direction features (1 for flow + 8 for directions = 9 features)
    features = np.hstack([flow.reshape(-1, 1), direction_encoded])

    # Create a dictionary to map times to indices
    time_to_index = {
        time: i for i, time in enumerate(df["15 Minutes"].str.split(" ").str[1])
    }

    # Find the index for the input time
    if time_input not in time_to_index:
        raise ValueError("Invalid time input. Please use the format 'HH:MM'.")

    index = time_to_index[time_input]

    # One-hot encode the input direction
    direction_categories = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
    if direction_input not in direction_categories:
        raise ValueError(
            f"Invalid direction input. Valid directions are {direction_categories}"
        )

    direction_onehot = encoder.transform([[direction_input]])

    # Prepare the input for prediction
    lags = 4
    if index < lags:
        raise ValueError("Not enough historical data for the given time.")

    # Extract the last `lags` timesteps of features (flow + direction)
    X_pred = features[index - lags : index].reshape(
        1, lags, 9
    )  # 9 features (flow + directions)

    # Overwrite the direction feature in the input with the one-hot encoded direction
    for i in range(lags):
        X_pred[0, i, 1:] = direction_onehot

    # Check if the model is SAES and flatten input only if it is
    if "saes" in model_path.lower():
        # Flatten the input for SAES model (expects 36 features)
        X_pred_flat = X_pred.reshape(1, -1)  # Flatten to (1, 36)
        # Make prediction for SAES
        predicted = model.predict(X_pred_flat)
    else:
        # Keep input shape as (1, 4, 9) for other models
        predicted = model.predict(X_pred)

    predicted = scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]

    return predicted


def predict_flow(scats_num, time, direction, model_type):
    model_path = MODEL_DIR + "/" + scats_num + "_" + model_type + ".keras"
    csv_path = CSV_DIR + "/" + scats_num + "_" + "trafficflow.csv"

    print(model_path)
    print(csv_path)
    predicted_flow = predict_traffic_flow(time, direction, model_path, csv_path)
    print(
        f"Predicted traffic flow at {time} in direction {direction}: {predicted_flow:.2f} vehicles per 15 minutes"
    )
    print("----------------------------------------")

    return predicted_flow


def main():
    # Load Keras models and predict traffic flow including directions
    for model_name in MODELS:
        model_path = f"./saved_models/{model_name}.keras"
        print(model_path)
        cpredict(model_path, TEST_CSV_DIRECTION)


def cpredict(model_path, data_path):
    model_name = get_model_name(model_path)

    print(f"-------------- {model_name} --------------")

    time_input = "11:30"  # Specify the time input for prediction
    direction_input = "W"  # Specify the direction input for prediction

    predicted_flow = predict_traffic_flow(
        time_input, direction_input, model_path, data_path
    )

    print(
        f"Predicted traffic flow at {time_input} in direction {direction_input}: {predicted_flow:.2f} vehicles per 15 minutes"
    )

    print("----------------------------------------")


def original_predict(model_path, train_csv):
    lags = 4

    model = load_model(model_path)
    print("Model loaded successfully!")

    X_train, y_train, scaler = data.original_process(train_csv, lags)

    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(1, -1)[0]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Make predictions
    predicted = model.predict(X_train)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    print("Size of y_train -> ", len(predicted))

    plot_results(y_train[:96], predicted[:96])

    # 96 -> 1 day
    print("Predicted 97 -> ", predicted[:96])
    print("Predicted Array -> ", predicted)


def get_model_name(model_path):
    return model_path.split("/")[-1].split(".")[0].upper()


if __name__ == "__main__":
    main()
