#!/usr/bin/env python

# Convert Celsius to Farenheit
# Author: Samuel Adamson
# Last Edit: 11/22/2021

import tensorflow as tf
import matplotlib.pyplot as plot
import training_data as td
import os


# Create and Train Model
# Return Type: Tuple
# @RETURN: ( training history, model )
def ct_model():
    # Model -- l_0
    #   -- Single Layer
    #   -- Input is a single value, 1d array
    l_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([l_0])

    # Configure model for training
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
    # Train Model
    history = model.fit(td.celsius, td.farenheit, epochs=1000, verbose=False)
    print("Model Training Complete!")

    return (history, model)


# Training Visualization
# Return Type: None
# @RETURN: None
def display_training(history):
    # Setup Plot
    plot.xlabel('Epoch')
    plot.ylabel('Loss')
    plot.plot(history.history['loss'])

    # # Save plot
    # plot.savefig('./figs/training.png')
    # Show Plot
    plot.show()


# Predict Value
# Return Type: None
# @RETURN: None
def predict(model):
    # Get Input From Terminal
    celsius_in = float(input('Enter celsius value for conversion: '))
    # Print Prediction
    print(model.predict([celsius_in]))


# Execution
if __name__ == '__main__':
    # Set logging for errors only
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Create and train model
    history, model = ct_model()

    # td.display_values()
    # display_training(history)
    # predict(model)