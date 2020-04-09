# coding: utf-8

# In[4]:


from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import os
import seaborn as sns
import warnings

np.random.seed(0)
from sklearn.model_selection import train_test_split

# In[5]:


from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

get_ipython().run_line_magic('matplotlib', 'inline')

# In[6]:


TRAIN_DATASET = 'train'
TEST_DATASET = 'test1'
IMG_SIZE = 50


# In[7]:


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return [0, 1]


# In[8]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DATASET)):
        label = label_img(img)
        path = os.path.join(TRAIN_DATASET, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_data


# In[9]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DATASET)):
        path = os.path.join(TEST_DATASET, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    # np.save('test_data.npy', testing_data)
    return testing_data


# In[ ]:


# 2. Preprocessing for the input images


# In[10]:


train_filenames = create_train_data()
test_filenames = process_test_data()

# In[11]:


print(train_filenames[0:10])
print(test_filenames[0:10])

# In[12]:


# Let’s see the total number of images in training set and testing set
train_cat = filter(lambda x: x.split(".")[0] == "cat", train_filenames)
train_dog = filter(lambda x: x.split(".")[0] == "dog", train_filenames)
x = ['train_cat', 'train_dog', 'tensorbroad_pb_android']
y = [len(train_cat), len(train_dog), len(test_filenames)]
ax = sns.barplot(x=x, y=y)

# In[ ]:


# Training sets were further divided into 90% for training the model and 10% for evaluate the model using cross validation.

my_train, my_cv = train_test_split(train_filenames, test_size=0.1, random_state=0)
print(len(my_train), len(my_cv))
