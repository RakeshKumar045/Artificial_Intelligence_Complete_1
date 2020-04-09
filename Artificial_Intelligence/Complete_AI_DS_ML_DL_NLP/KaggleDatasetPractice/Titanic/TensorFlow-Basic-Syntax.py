# coding: utf-8

# # TensorFlow_D Basics

# In[1]:


import tensorflow as tf

# In[2]:


# Make sure you are using 1.3 for exact sytnax matching!
print(tf.__version__)

# ## Tensors

# In[3]:


hello = tf.constant('Hello')

# In[4]:


type(hello)

# In[5]:


world = tf.constant('World')

# In[6]:


result = hello + world

# In[7]:


result

# In[8]:


type(result)

# In[9]:


with tf.Session() as sess:
    result = sess.run(hello + world)

# In[10]:


result

# ** Computations **

# In[11]:


tensor_1 = tf.constant(1)
tensor_2 = tf.constant(2)

# In[12]:


type(tensor_1)

# In[13]:


tensor_1 + tensor_2

# In[14]:


sess

# In[15]:


sess.close()

# ## Operations

# In[16]:


const = tf.constant(10)

# In[17]:


fill_mat = tf.fill((4, 4), 10)

# In[18]:


myzeros = tf.zeros((4, 4))

# In[19]:


myones = tf.ones((4, 4))

# In[20]:


myrandn = tf.random_normal((4, 4), mean=0, stddev=0.5)

# In[21]:


myrandu = tf.random_uniform((4, 4), minval=0, maxval=1)

# In[22]:


my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]

# ## Interactive Session
# 
# Useful for Notebook Sessions

# In[23]:


# Only run this cell once!
sess = tf.InteractiveSession()

# In[24]:


for op in my_ops:
    print(op.eval())
    print('\n')

# ## Matrix Multiplication

# In[58]:


a = tf.constant([[1, 2],
                 [3, 4]])

# In[59]:


a.get_shape()

# In[60]:


b = tf.constant([[10], [100]])

# In[61]:


b.get_shape()

# In[62]:


result = tf.matmul(a, b)

# In[63]:


result.eval()
