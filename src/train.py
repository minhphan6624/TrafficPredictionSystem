import sys

sys.dont_write_bytecode = True

import os
import warnings
import argparse
import numpy as np
import pandas as pd


from keras.models import Model
from keras.callbacks import EarlyStopping
from pathlib import Path


from training.model import get_lstm, get_gru, get_saes, get_cnn
from training.data import original_process


warnings.filterwarnings("ignore")

# Hyperparameters
EPOCHS = 300
BATCH_SIZE = 256
LAG = 4
SCATS_CSV_DIR = "../training_data/traffic_flows"
TEST_CSV = f"{SCATS_CSV_DIR}/970_N_trafficflow.csv"
SCATS_CSV_DIR_DIRECTION = "../training_data/new_traffic_flows"
TEST_CSV_DIRECTION = f"{SCATS_CSV_DIR_DIRECTION}/970_trafficflow.csv"

# Models with input shape reflecting 9 features (1 for flow + 8 for direction)
MODELS = {
    "lstm": get_lstm([LAG, 64, 64, 1]),  # 9 features total
    "gru": get_gru([LAG, 64, 64, 1]),
    "saes": get_saes([LAG, 128, 64, 32, 1]),
    "cnn": get_cnn([LAG, 128, 1]),
}


def get_early_stopping_callback():
    return EarlyStopping(
        monitor="loss",
        patience=50,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )


def train_model(model, X_train, y_train, name, config, print_loss):
    model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

    # Set up EarlyStopping callback
    model_path = "./saved_models/" + str(name) + ".keras"
    model_loss_path = "./saved_models/" + name + "_loss.csv"

    # Train the model with EarlyStopping
    hist = model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[get_early_stopping_callback()],
    )

    # if model exists, delete
    if os.path.exists(model_path):
        os.remove(model_path)

    # Save loss history if requested
    if print_loss:
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv(model_loss_path, encoding="utf-8", index=False)

    # Save the final trained model
    model.save(model_path)


def train_saes(models, X_train, y_train, name, config, print_loss):
    # Flatten the X_train for the SAES model
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the input

    temp = X_train_flat
    for i in range(len(models) - 1):
        if i > 0:
            prev_model = models[i - 1]

            input_tensor = prev_model.layers[0].input
            hidden_layer_output = prev_model.get_layer("hidden").output

            hidden_layer_model = Model(inputs=input_tensor, outputs=hidden_layer_output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

        m.fit(
            temp,
            y_train,
            batch_size=config["batch"],
            epochs=config["epochs"],
            validation_split=0.05,
            callbacks=[get_early_stopping_callback()],
        )

        models[i] = m

    # Train the final SAES model
    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer("hidden").get_weights()
        saes.get_layer("hidden%d" % (i + 1)).set_weights(weights)

    train_model(saes, X_train_flat, y_train, name, config, print_loss)


def train_models(model_types, model_prefix, csv, print_loss):
    config = {"batch": BATCH_SIZE, "epochs": EPOCHS}

    # Process data including direction
    X_train, y_train, _, _ = original_process(csv, LAG)

    # Reshape data to accommodate the multi-dimensional input (flow + direction)
    num_features = X_train.shape[2]  # Determine number of features dynamically
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

    # Flatten the features for SAES model
    X_train_saes = np.reshape(X_train, (X_train.shape[0], -1))

    for model_type in model_types:
        model_name = (
            model_prefix + model_type if model_prefix != None or "" else model_type
        )

        model_instance = MODELS.get(model_type)

        if model_type == "saes":
            train_saes(
                model_instance, X_train_saes, y_train, model_name, config, print_loss
            )
        else:
            train_model(
                model_instance, X_train, y_train, model_name, config, print_loss
            )


def train_scats(model_types):
    for path in Path(SCATS_CSV_DIR_DIRECTION).iterdir():
        if path.is_file():
            name = Path(path).name
            scats_data = name.split("_")
            scats_number = scats_data[0]
            print(f"------------  SCATS site: {scats_number}  ------------")

            model_prefix = str.format(
                "{0}_",
                scats_number,
            )
            print(model_types)
            train_models(model_types, model_prefix, path, False)


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        help="Model names (e.g. lstm gru tcn)",
        nargs="+",  # This allows multiple model names
        default=["lstm"],  # Default to a list containing "lstm"
    )

    # Add scats argument
    parser.add_argument(
        "--scats", help="Check if --scats is present", action="store_true"
    )

    parser.add_argument("--loss", help="Save loss history", action="store_true")

    # Parse the arguments
    args = parser.parse_args()

    if args.scats:
        train_scats(args.model)
    else:
        train_models(args.model, None, TEST_CSV_DIRECTION, args.loss)


if __name__ == "__main__":
    main(sys.argv)
