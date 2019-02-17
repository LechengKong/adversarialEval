
# coding: utf-8
# Author: Jerry Kong     jerry.kong@wustl.edu
# Usage: copy into jupyter notebook by order or run as python script
# Note: Model structure copied from tensorflow official website
# In[139]:

#import library
import tensorflow as tf     #cpu tensorflow   (may work on gpu)
from tensorflow import keras        #use keras as frontend
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


# In[2]:

#load keras mnist dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[3]:

#generate flattened dataset (not used in training)
flat_train = train_images.reshape((60000,28*28))
flat_test = test_images.reshape((10000,28*28))


# In[4]:

#data set validation
print(train_images.shape)
print(flat_train.shape)


# In[5]:

#classname enumeration
classname = ['0','1','2','3','4','5','6','7','8','9']


# In[6]:

#check the first training data
plt.imshow(train_images[0], cmap="gray")
plt.show()
train_labels[0]


# In[7]:

#normalize the dataset to range [0, 1]
train_images = train_images/255.0
test_images = test_images/255.0


# In[151]:

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


# In[152]:

#checkpoint saving callback initialization   (address format may change based on different operating system)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# In[153]:

#create model and fit on the traning dataset
model = createModel()
model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])


# In[342]:

#set constraint for each pixel value to [0, 1]
constr = [[0,1]]*784


# In[343]:

# func:
# Function to calculate the loss and gradient of the loss
# It encodes the optimization problem:   Minimize  c*norm2|x'-x| + loss(x', l)
# Where x' is the adversarial sample, and x is the original sample, l is the desired label
# x, x' is within [0, 1], the constraint is enforced by the scipy minimize function
# This function should be able to meet the need of our whole experiment
# Parameter:
# x: the image pixel value as a flatten nparray
# c: the coefficient that balances the effect of distortion term and loss term
# n: the index of the image that is used to generate adversarial sample   (crapy implementation need to be changed into batch or other implementation)
# sess: the tensorflow sess that is being used to evaluate the loss and gradient
# loss: the loss tensor
# grad: the gradient tensor

# Return:
# y[0]: y is an array with one single variable, y[0] is the loss
# gra: the gradient evaluated at x

def func(x, c, n, sess, loss, grad):
    ad = x.reshape((1,28,28))                      #reshape x to pass image into the network
    grad_val = sess.run(grad,feed_dict={intensor: ad})
    loss_val = loss.eval(feed_dict={intensor: ad},session = sess)
    y = np.linalg.norm(ad-test_images[n])*c + loss_val
    gra = (np.array(grad_val, dtype = np.float64)+c*(2*ad[0]-2*test_images[n])).flatten()
    return y[0], gra


# In[366]:

# get current tf session and initialize the model to evaluate loss and gradient
sess = tf.Session()
keras.backend.set_session(sess)
tmodel = createModel()
tmodel.load_weights(checkpoint_path)
testensor = tf.convert_to_tensor([0])   # crapy crapy shity code, this line tells minimizer what adversarial label we want, 0 means 0, 1 means 1, etc.

#tensor construct
intensor = tmodel.input
outtensor = tmodel.output
lo = keras.losses.sparse_categorical_crossentropy(testensor,outtensor)
g = tf.gradients(lo, intensor)

# pick c and n
c = 0.01
n = 1000

#noise term to be add on to the original image (this is not a must, actually don't add it)
noise = np.random.uniform(size=(28,28))/10
#actual minimization function , see scipy minimize reference for details
ad = op.minimize(func, test_images[n].flatten(), method = 'L-BFGS-B', jac = True,args=(c, n, sess, lo, g), bounds = constr)


# In[367]:

# adversarial sample generation evaluation
print(ad)
asample = ad.x.reshape((1,28,28))
print((asample-test_images[2]).sum())
pic = asample[0]*255
plt.imshow(pic, cmap="gray")
plt.show()
print(test_labels[n])
print(tmodel.predict(asample))


# In[136]:

# its ok if we use this line to save the model weight instead of using a callback function
# Actually when we get to more complicated model this is probably what we want to go for
# model.save_weights('./checkpoints/line')


# In[129]:

# plot the model, if error, follow the error to install the essential pacakges
keras.utils.plot_model(model, to_file='model.png')


# In[245]:

# when quiting the script, make sure the session is closed, so it does not suck up our your computational power
sess.close()

