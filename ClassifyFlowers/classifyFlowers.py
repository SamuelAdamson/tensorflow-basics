#!/usr/bin/env python

# Author: Samuel Adamson
# Last Edited: 02/06/2022
# Classifying flowers using CNN

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
    # Random seed generator
    rng = tf.random.Generator.from_seed(123, alg='philox')

    # Normalize data for training
    # PARAMS: Image
    # RETURN: Normalized image
    def _normalize(image, label):
        # Normalize
        image = image.tf.cast(image, tf.float32) / 255.0

        return image, label

    # Augment training data
    # PARAMS: Training image, seed generator
    # RETURN: Augmented Training image
    def _augment(image, label, seed_gen=rng):
        # Create seed
        seed = seed_gen.make_seeds(2)[0]

        # Normalize image
        image, label = _normalize(image, label)

        # Random flip left-right
        image = tf.image.random_flip_left_right(image, seed=seed)
        #Random flip up-down
        image = tf.image.random_flip_up_down(image, seed=seed)
        # Random Brightness
        image = tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed)

        return image, label

    # Training dataset - 80% of images
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    # Validation dataset - 80% of images
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    # Preprocess training dataset
    #   Normalize and Augment
    train_ds = (train_ds
        .shuffle(1000)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)        
    )

    # Preprocess validation dataset
    #   Normalize
    valid_ds = (valid_ds
        .map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return (train_ds, valid_ds)


# Visualize some test data



# Program entry point
if __name__ == '__main__':
    # Get and preprocess data
    base_dir = getData()
    train_ds, valid_ds = preprocessData(base_dir)


