import tensorflow as tf
from tensorflow.keras import layers

import numpy as np 

def loadDataset():
    (X, y), _= tf.keras.datasets.fashion_mnist.load_data()
    X_train,y_train,X_valid,y_valid = X[:50000].astype('float32'),y[:50000],X[50000:].astype('float32'),y[50000:]
    X_train/=255
    X_valid/=255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    X_train = np.expand_dims(X_train, -1)
    X_valid = np.expand_dims(X_valid, -1)

    return X_train,y_train,X_valid,y_valid