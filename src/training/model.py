from keras.layers import (
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    Flatten,
    SimpleRNN,
    Dense,
    Dropout,
    Activation,
)
from keras.models import Sequential
from tcn import TCN
import tensorflow as tf


# Define LSTM model with 9 features
def get_lstm(units):
    model = Sequential()
    # Update input_shape to handle 9 features (1 traffic flow + 8 direction features)
    model.add(
        LSTM(units[1], input_shape=(units[0], 9), return_sequences=True)
    )  # 9 features
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation="sigmoid"))

    return model


# Define GRU model with 9 features
def get_gru(units):
    model = Sequential()
    # Update input_shape to handle 9 features
    model.add(
        GRU(units[1], input_shape=(units[0], 9), return_sequences=True)
    )  # 9 features
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation="sigmoid"))

    return model


def _get_sae(inputs, hidden, output):
    model = Sequential()
    model.add(Dense(hidden, input_shape=(inputs,), name="hidden"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation="sigmoid"))

    return model


def get_saes(layers):
    # Adjust for input size of 36 (4 timesteps * 9 features)
    sae1 = _get_sae(36, layers[1], layers[-1])  # Input is now 36 features
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_shape=(36,), name="hidden1"))  # 36 input features
    saes.add(Activation("sigmoid"))
    saes.add(Dense(layers[2], name="hidden2"))
    saes.add(Activation("sigmoid"))
    saes.add(Dense(layers[3], name="hidden3"))
    saes.add(Activation("sigmoid"))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation="sigmoid"))

    models = [sae1, sae2, sae3, saes]

    return models


# Define CNN model with 9 features
def get_cnn(units):
    model = Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(units[0], 9),
        )
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units[1], activation="relu"))
    model.add(Dense(units[2], activation="sigmoid"))  # Or 'linear' for regression
    return model
