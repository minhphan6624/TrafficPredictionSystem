import sys
sys.dont_write_bytecode = True

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data import original_process
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_results(y_true, y_pred):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """

    d = '2016-10-1 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred, label='LSTM')
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def predict_traffic_flow(time_input, model_path, data_path):
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess the data
    df = pd.read_csv(data_path, encoding='utf-8').fillna(0)
    attr = 'Lane 1 Flow (Veh/15 Minutes)'
    
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
    flow = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    
    # Create a dictionary to map times to indices
    time_to_index = {time: i for i, time in enumerate(df['15 Minutes'].str.split(' ').str[1])}
    
    print(time_to_index)

    # Find the index for the input time
    if time_input not in time_to_index:
        raise ValueError("Invalid time input. Please use the format 'HH:MM'.")
    
    index = time_to_index[time_input]
    
    # Prepare the input for prediction
    lags = 4
    if index < lags:
        raise ValueError("Not enough historical data for the given time.")
    
    X_pred = flow[index - lags: index].reshape(1, lags, 1)
    
    # Make prediction
    predicted = model.predict(X_pred)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]
    
    return predicted

def main():
    # Load in keras model
    train_csv = '../../data/traffic_flows/970_E_trafficflow.csv'

    cpredict("saved_models/lstm.keras", train_csv)

    #original_predict(train_csv)

def cpredict(model_path, data_path):
    time_input = "08:30"
    predicted_flow = predict_traffic_flow(time_input, model_path, data_path)
    print(f"Predicted traffic flow at {time_input}: {predicted_flow:.2f} vehicles per 15 minutes")

def original_predict(train_csv):
    lags = 4

    model = load_model("saved_models/lstm.keras")
    print("Model loaded successfully!")

    X_train, y_train, scaler = original_process(train_csv, lags)

    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(1, -1)[0]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Make predictions
    predicted = model.predict(X_train)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    print("Size of y_train -> ", len(predicted))

    plot_results(y_train[: 96], predicted[:96])

    # 96 -> 1 day
    print("Predicted 97 -> ", predicted[:96])
    print("Predicted Array -> ", predicted)

if __name__ == "__main__":
    main()