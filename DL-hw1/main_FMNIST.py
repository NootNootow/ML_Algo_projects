import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import KFold

from tensorflow.keras import layers
from util_FMNIST import loadDataset
from model_FMNIST import get_model

#loading Data and spliting training data set into train and test
print("Loading Data")
X_train,y_train,X_test,y_test = loadDataset() 

print("X_train, y_train", X_train.shape,y_train.shape)
print("X_test, y_test", X_test.shape,y_test.shape)
#Declaring variables 
input_shape = X_train[0].shape
num_classes = 10
batch_size = 32
epochs = 20

#Declaring early stopping 
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

filepath = "epochs/epochs_{epoch:03d}"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True)

cb=[early_stopping,checkpoint]

# Architect-1 (CNN without ResBlock)
#Run 1 - 4 
print()
print("Architect-1")
print("CNN without Resblock")
print()
print("Regularizers + 1st set of parameters")
print("RUN - 1 ")
print()
model= get_model(input_shape, num_classes, units=[32,64], kernel_size=(2,2),padding='valid', activation="relu",
					kernel_regularizer=tf.keras.regularizers.l2(1e-3),use_ResBlock=False)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print()
print("Regularizers + 2nd set of parameters")
print("RUN - 2 ")
print()
model= get_model(input_shape, num_classes, units=[64,128], kernel_size=(3,3),padding='same', activation="selu",
					kernel_regularizer=tf.keras.regularizers.l2(1e-3),use_ResBlock=False)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print()
print("No regularizers + 1st set of parameters")
print("RUN - 3 ")
print()
model= get_model(input_shape, num_classes, units=[32,64], kernel_size=(2,2),padding='valid', activation="relu",
				use_ResBlock=False)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print()
print("No regularizers + 2nd set of parameters")
print("RUN - 4 ")
print()
model= get_model(input_shape, num_classes, units=[64,128], kernel_size=(3,3),padding='same', activation="selu",
			use_ResBlock=False)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")

# Architect-2 (CNN with ResBlock)
#Run 5 - 8 
print()
print("Architect-2")
print("CNN with ResBlock")
print()
print("Regularizers + 1st set of parameters")
print("RUN - 5 ")
print()
model= get_model(input_shape, num_classes, units=[32,64], kernel_size=(2,2),padding='valid', activation="relu",
					kernel_regularizer=tf.keras.regularizers.l2(1e-3),use_ResBlock=True)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print()
print("Regularizers + 2nd set of parameters")
print("RUN - 6 ")
print()
model= get_model(input_shape, num_classes, units=[64,128], kernel_size=(3,3),padding='same', activation="selu",
					kernel_regularizer=tf.keras.regularizers.l2(1e-3),use_ResBlock=True)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print()
print("No regularizers + 1st set of parameters")
print("RUN - 7 ")
print()
model= get_model(input_shape, num_classes, units=[32,64], kernel_size=(2,2),padding='valid', activation="relu",
				use_ResBlock=True)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
print()
print("No regularizers + 2nd set of parameters")
print("RUN - 8 ")
print()
model= get_model(input_shape, num_classes, units=[64,128], kernel_size=(3,3),padding='same', activation="selu",
			use_ResBlock=True)

tf.saved_model.save(model,"save_model/")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.10, callbacks=cb)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("----------------------------------------------------------------------------------")
print(tf.math.confusion_matrix(
    np.argmax(y_test,axis=1), np.argmax(model.predict(X_test),axis=1),
))
print("----------------------------------------------------------------------------------")
