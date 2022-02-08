#!/usr/bin/env python

# Author: Samuel Adamson
# Last Edited: 02/06/2022
# Classifying flowers using CNN

from distutils.command.build import build
import os
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt


# Retrieve and extract data
# PARAMS: None
# RETURN: File dir w/ data (string), Classes of flowers (list)
def getData():
    # Get data from URL, read in to local directory
    DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    zip_file = tf.keras.utils.get_file(origin=DATA_URL, fname="flower_photos.tgz", extract=True, cache_dir='./')
    
    # Base directory
    base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
    
    return base_dir



# Preprocess data - prepare for training and validation
# PARAMS: Training Data Path, Validation Data Path, Batch, Img Dimensions
# RETURN: Augmented Training Dataset, Validation Dataset
def preprocessData(data_dir, batch_size=100, img_size=150):

    # Normalize data for training
    # PARAMS: Image
    # RETURN: Normalized image
    def _normalize(image, label):
        # Normalize
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    # Augment training data
    # PARAMS: Training image, seed generator
    # RETURN: Augmented Training image
    def _augment(image, label):
        # Normalize image
        image, label = _normalize(image, label)

        # Random flip horizontally and vertically
        image = tf.image.random_flip_left_right(image, seed=np.random.randint(100))
        image = tf.image.random_flip_up_down(image, seed=np.random.randint(100))
        # Random contrast and saturation
        image = tf.image.random_contrast(image, lower=0.8, upper=1.0, seed=np.random.randint(100))
        image = tf.image.random_saturation(image, lower=0.8, upper=1.0, seed=np.random.randint(100))

        return image, label

    # Training dataset - 80% of images
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=1
    )

    # Validation dataset - 80% of images
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=1
    )

    # Get class names
    class_names = valid_ds.class_names

    # Preprocess training dataset
    #   Normalize and Augment
    train_ds = train_ds.map(_augment)

    # Preprocess validation dataset
    #   Normalize
    valid_ds = valid_ds.map(_normalize)

    return train_ds, valid_ds, class_names


# Visualize some data
# PARAMS: Tensorflow dataset Length:20, Output file path
# RETURN: None
def visualize(dataset, filepath, class_names):
    # Format plot figure size
    plt.figure(figsize=(12,12))

    # Iterate through subsidiary data
    for images, labels in dataset:
        for i in range(25):
            # Plot
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[labels[i]])

    # Save plot
    # !mkdir figs
    plt.savefig(filepath)
    # Show Plot
    # plt.show()


# Build model
#   Three 2D Convolutional Layers
#   Three Max Pooling Layers Following Convoltions
#   Flatten Layer 
#   Densely connected layer with 512 Units
#   20% Dropout
# PARAMS: Image size
# RETURN: Untrained Model
def buildModel(img_size=150):
    # Define model
    model = tf.keras.models.Sequential([
        # Convolutional Block 16 filters
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        # Convolutional Block 32 filters
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Convolutional Block 32 filters
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten, Dropout, Dense Layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        # Dropout, and softmax activated dense for 5 classes
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

    # Show model summary
    model.summary()

    # Return untrained model
    return model


# Train given model
# PARAMS: Untrained Model, Training Dataset, Validation Dataset, Visual File Path, Epochs, Batch size
# RETURN: Trained Model
def trainModel(model, train_ds, valid_ds, filepath, _epochs=80, _batch_size=100):

    # Store training
    history = model.fit(
        train_ds, batch_size=_batch_size, epochs=_epochs,
        validation_data=valid_ds
    )

    # Store Accuracy
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    # Store Loss
    loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Range of epochs
    epochs_range = range(_epochs)

    # Plot configuration -- Accuracy
    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Plot configuration -- Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    # Save figure
    plt.savefig(filepath)
    plt.show()

    # Return trained model
    return model



# Program entry point
if __name__ == '__main__':
    # Get and preprocess data
    base_dir = getData()
    train_ds, valid_ds, class_names = preprocessData(base_dir)

    # Visualize some sample data
    visualize(train_ds.take(1), './figs/training_sample.png', class_names)
    visualize(valid_ds.take(1), './figs/validation_sample.png', class_names)

    # Get model
    model = buildModel()

    # Train model
    trained_model = trainModel(model, train_ds, valid_ds, './figs/training.png')

    # !mkdir ./model
    trained_model.save('./model')