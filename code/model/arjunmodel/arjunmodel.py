import tensorflow as tf
import numpy as np
import pandas as pd
from load_and_preprocess import preprocess_features

"""
**************************************************
This file compiles, trains, and tests the model  *
before performing inference on recent NBA games. *
**************************************************
"""


def create_model():
    # get features, train, and test
    all_features, all_inputs, train_ds, test_ds = preprocess_features()

    # Add layers
    x = tf.keras.layers.Dense(193 * 10, activation="relu")(all_features)
    x = tf.keras.layers.Dense(193 * 8, activation="relu")(x)
    x = tf.keras.layers.Dense(193 * 6, activation="relu")(x)
    x = tf.keras.layers.Dense(193 * 4, activation="relu")(x)
    x = tf.keras.layers.Dense(193 * 2, activation="relu")(x)
    x = tf.keras.layers.Dense(193 * 1, activation="relu")(x)
    x = tf.keras.layers.Dense(int(193 * .5), activation="relu")(x)
    x = tf.keras.layers.Dense(int(193 * .25), activation="relu")(x)
    x = tf.keras.layers.Dense(int(193 * .1), activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),  # what loss function?
                  metrics=[tf.keras.metrics.MeanSquaredError()])  # what metric?

    tf.keras.utils.plot_model(model, to_file="./model_architecture.png", show_shapes=True, rankdir="LR")
    return model, train_ds, test_ds


def train_and_testmodel(model, train_ds, test_ds, name_to_save='nba'):
    model.fit(train_ds, epochs=40)
    model.save(name_to_save)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    return name_to_save

def predict(model_name):
    model = tf.keras.models.load_model(model_name)
    data = pd.read_csv("/Users/arjunshanmugam/Desktop/Untitled.csv")
    row1 = data.iloc[0]
    print(row1.to_dict())

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in row1.items()}
    predictions = model.predict(input_dict)
    print(predictions)