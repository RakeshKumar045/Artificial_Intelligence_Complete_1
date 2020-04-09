# coding: utf-8

# In[18]:


import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays

mnist = tf.keras.datasets.mnist  # mnist is a dataset_D of 28x28 images of handwritten digits and their labels
(x_train, y_train), (
    x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

print(tf.__version__)

# In[30]:


# print(x_train[0])

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# In[6]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# In[9]:


print(y_train[0])
print(y_test[0])

# In[10]:


# print(x_train[0])


# In[19]:


# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[20]:


# print(x_train[0])


# In[14]:


plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# In[15]:


plt.imshow(x_train[0])
plt.show()

# In[21]:


model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(
    tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(
    tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10,
                                activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',
              # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

# In[31]:


model.fit(x_train, y_train, epochs=10)  # train the model

# In[32]:


val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

# In[33]:


model.save('epic_num_reader.model')  # save model

# In[34]:


new_model = tf.keras.models.load_model('epic_num_reader.model')  # load model

# In[35]:


# finally, make predictions!

predictions = new_model.predict(x_test)
print(predictions)

# In[36]:


import numpy as np

print(np.argmax(predictions[0]))

# In[37]:


# There's your prediction, let's look at the input:

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()

# In[38]:


plt.imshow(x_test[0])
plt.show()
