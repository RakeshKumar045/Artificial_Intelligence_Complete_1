# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# In[2]:


simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']

# In[3]:


# instantiate CountVectorizer (vectorizer)
# TfidfVectorizer() is better than CountVectorizer() 
# CountVectorizer()  is almost same as TfidfVectorizer(),CountVectorizer() is having less feature than TfidfVectorizer()
vect = CountVectorizer()

# <h3>Learn the 'vocabulary' of the training data (occurs in-place)</h3>

# In[4]:


vect.fit(simple_train)

# In[5]:


simple_train

# In[6]:


# examine the fitted vocabulary
# get uniuqe name
vect.get_feature_names()

# <h3>Transform training data into a 'document-term matrix'</h3>

# In[7]:


simple_train_dtm = vect.transform(simple_train)
print(simple_train)
print(vect.get_feature_names())
simple_train_dtm.toarray()

# <h3>Convert sparse matrix to a dense matrix</h3>

# In[8]:


simple_train = ['call call please please', 'call you tonight', 'Call me a cab', 'please call me.. please']
vect = CountVectorizer()
vect.fit(simple_train)
simple_train_dtm = vect.transform(simple_train)
simple_train

# In[9]:


vect.get_feature_names()

# In[17]:


simple_train_dtm.toarray()

# <h3>Examine the vocabulary and document-term matrix together</h3>

# In[18]:


# pd.DataFrame(matrix, columns=columns)
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
