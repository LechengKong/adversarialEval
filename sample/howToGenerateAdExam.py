# pre-req: cfinder.py in the same directory

#import library
import tensorflow as tf     #cpu tensorflow   (may work on gpu)
from tensorflow import keras        #use keras as frontend
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from cfinder import cfind


#load keras mnist dataset  (or any data set you like)
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#generate flattened dataset (not used in training, code optimize model to incoporate this, could be a big performance boost)
flat_train = train_images.reshape((60000,28*28))
flat_test = test_images.reshape((10000,28*28))


#normalize the dataset to range [0, 1]
train_images = train_images/255.0
test_images = test_images/255.0

#simple fully-connected neural network creation
def createModel():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


#checkpoint saving callback initialization   (address format may change based on different operating system)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


#create model and fit on the traning dataset
model = createModel()
model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])


#create cfinding object
#only using the first 10 samples to save time
cf = cfind(test_images[0:11], test_labels[0:11], createModel(), checkpoint_path)

cf.test_initialize()

cf.findAd()

#access the samples and c values, you can either save them or directly use them
samples = cf.getAdvSample()

cs = cf.getC()

print(cs)