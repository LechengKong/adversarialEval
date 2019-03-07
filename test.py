#import library
import tensorflow as tf     #cpu tensorflow   (may work on gpu)
from tensorflow import keras        #use keras as frontend
from keras import regularizers
from keras.callbacks import EarlyStopping
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True' # this is for mac OS, ingore this if you are using windows 
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

#to change the type to float which could make the learning process easier
train_images=train_images.astype('float32')
test_images=test_images.astype('float32')

def createModel():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(200, activation=tf.nn.sigmoid,kernel_regularizer=regularizers.l2(0.00000005)),
        keras.layers.Dense(200, activation=tf.nn.sigmoid,kernel_regularizer=regularizers.l2(0.00000005)),
        keras.layers.Dense(10, activation=tf.nn.sigmoid,kernel_regularizer=regularizers.l2(0.0000001)),
    ])
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

ad = np.load('FC(2x2x1x-7).npy')
model = createModel()
model.load_weights("training_1/cp.ckpt")
lenth=len(ad)
score=model.evaluate(ad,test_labels[0:lenth],verbose=0)
print('Test Error on ADV',1-score[1])

