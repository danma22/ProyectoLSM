"""
Title: Timeseries classification from scratch
Author: [hfawaz](https://github.com/hfawaz/)
Date created: 2020/07/21
Last modified: 2021/07/16
Description: Training a timeseries classifier from scratch on the FordA dataset from the UCR/UEA archive.
"""
"""
## Introduction
This example shows how to do timeseries classification from scratch, starting from raw
CSV timeseries files on disk. We demonstrate the workflow on the FordA dataset from the
[UCR/UEA archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).
"""

# Modified and used for the Mexican Sign Language dataset
# Modified by Agustin Zavala, Jose Avalos, Roberto Higuera, Jesus Quiñones

"""
## Setup
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Configuración de GPU
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# print tensorflow version
print(tf.__version__)

from .data_loader import load_data_numpy


def train_nn():

    x_train, y_train = load_data_numpy("../dataset")
    x_test, y_test = load_data_numpy("../dataset")

    print(type(x_train), type(y_train))

    """
    ## Visualize the data
    Here we visualize one timeseries example for each class in the dataset.
    """

    classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    plt.figure()
    for c in classes:
        c_x_train = x_train[y_train == c]
        plt.plot(c_x_train[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()

    """
    ## Standardize the data
    Our timeseries are already in a single length (500). However, their values are
    usually in various ranges. This is not ideal for a neural network;
    in general we should seek to make the input values normalized.
    For this specific dataset, the data is already z-normalized: each timeseries sample
    has a mean equal to zero and a standard deviation equal to one. This type of
    normalization is very common for timeseries classification problems, see
    [Bagnall et al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).
    Note that the timeseries data used here are univariate, meaning we only have one channel
    per timeseries example.
    We will therefore transform the timeseries into a multivariate one with one channel
    using a simple reshaping via numpy.
    This will allow us to construct a model that is easily applicable to multivariate time
    series.
    """

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    """
    Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
    the number of classes beforehand.
    """

    num_classes = len(np.unique(y_train))

    """
    Now we shuffle the training set because we will be using the `validation_split` option
    later when training.
    """

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    """
    Standardize the labels to positive integers.
    The expected labels will then be 0 and 1.
    """

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    """
    ## Build a model
    We build a Fully Convolutional Neural Network originally proposed in
    [this paper](https://arxiv.org/abs/1611.06455).
    The implementation is based on the TF 2 version provided
    [here](https://github.com/hfawaz/dl-4-tsc/).
    The following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found
    via random search using [KerasTuner](https://github.com/keras-team/keras-tuner).
    """

    def make_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(
            input_shape=input_shape, filters=128, kernel_size=3, padding="same"
        )(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    model = make_model(input_shape=x_train.shape[1:])
    print(model.summary())
    # keras.utils.plot_model(model, show_shapes=True)

    """
    ## Train the model
    """

    epochs = 150
    batch_size = 8

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "../best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    """
    ## Evaluate model on test data
    """

    model = keras.models.load_model("../../best_model.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    """
    ## Plot the model's training and validation loss
    """

    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    """
    We can see how the training accuracy reaches almost 0.95 after 100 epochs.
    However, by observing the validation accuracy we can see how the network still needs
    training until it reaches almost 0.97 for both the validation and the training accuracy
    after 200 epochs. Beyond the 200th epoch, if we continue on training, the validation
    accuracy will start decreasing while the training accuracy will continue on increasing:
    the model starts overfitting.
    """


if __name__ == "__main__":
    train_nn()
