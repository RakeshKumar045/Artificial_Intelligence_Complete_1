# coding: utf-8

# In[32]:


import pandas as pd
from collections import Counter as c
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# In[6]:


data = pd.read_csv("abhilasha_dataset_updated.csv", encoding="ISO-8859-1")

# In[7]:


data.shape

# In[8]:


data.columns

# In[9]:


data.head(2)

# In[10]:


data.isnull().sum()

# In[11]:


data.dropna(subset=['category'], how='all', inplace=True)

# In[12]:


data.isnull().sum()

# In[13]:


cleanup_nums = {"category": {"ignore": 1
    , "decision": 2
    , "action point": 3
    , "key point": 4
    , "key notes": 5
    , "ingore": 6
    , "key points": 7
    , "action plan": 8
                             }}

# In[14]:


c(data.category)

# In[15]:


enc = LabelEncoder()

# In[16]:


# enc=OneHotEncoder(sparse=False)


# In[17]:


data["category_label"] = enc.fit_transform(data["category"])

# In[18]:


# data["category_label"]


# In[19]:


data[["category_label", "category"]].head(10)

# In[20]:


# data.replace(cleanup_nums, inplace=True)


# In[21]:


type(data.category)

# In[22]:


x = data.loc[:, ["description"]]
y = data["category_label"]

# In[23]:


a = data.iloc[:, 0]
b = data.iloc[:, -1]

# In[24]:


type(y)

# In[25]:


data.dtypes

# In[26]:


# b


# In[33]:


# model = LogisticRegression()

# Instantiate the classifier
model = GaussianNB()

# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x, b, test_size=0.3, random_state=1)

# In[35]:


model.fit(x_train, y_train)

# In[31]:


# y_trai


# In[67]:


c(data.category)

# In[36]:


# Instantiate the classifier
gnb = GaussianNB()

# In[38]:


# x


# In[39]:


c(data.description)
