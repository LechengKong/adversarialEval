# pre-req: cfinder.py in the same directory

#import library
import tensorflow as tf     #cpu tensorflow   (may work on gpu)
from tensorflow import keras        #use keras as frontend
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True' # this is for mac OS, ingore this if you are using windows 
from cfinder import cfind

# num is the number of the adversrail example that we intend to generate, by default is 10000
num=10000
Trainingtimes=80
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


#simple fully-connected neural network creation
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

    
      
    
#checkpoint saving callback initialization   (address format may change based on different operating system)
checkpoint_path = "sigmoid200/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#here the best practice is not to train the model every time, but train the model and save the directory
#pass directory to cfind object to get model weight
#create model and fit on the traning dataset
model = createModel()
overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 64)
model.fit(train_images, train_labels, epochs=Trainingtimes, callbacks = [overfitCallback,cp_callback])
score_tr=model.evaluate(train_images, train_labels,verbose=0)
print('Training Error:',1-score_tr[1])
score_te=model.evaluate(test_images, test_labels,verbose=0)
print('Testing Error:',1-score_te[1])

#create cfinding object
#only using the first 10 samples to save time
cf = cfind(test_images[0:num], test_labels[0:num], createModel(), checkpoint_path)

cf.test_initialize()

cf.findAd()

#access the samples and c values, you can either save them or directly use them
samples = cf.getAdvSample()
np.save('FC(2x2x1x-7)',samples)
model.load_weights(checkpoint_path)
score=model.evaluate(samples,test_labels[0:num],verbose=0)
print(score)
c = cf.getC()
sh=(samples-test_images[0:num]).reshape((num,784))
en=np.matmul(sh,np.transpose(sh))/784
f=np.sum(np.sqrt(en.diagonal()))/num
print(f)
#print('this is C',c)
