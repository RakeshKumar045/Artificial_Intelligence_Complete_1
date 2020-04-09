# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None)

# In[3]:


data.head(1)

# In[4]:


data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'outcome']

# In[5]:


data.head(1)

# In[28]:


from collections import Counter as c

c(data.outcome)

# In[29]:


# data.persons.replace('more','5',inplace=True)


# In[30]:


c(data.buying)

# In[31]:


c(data.maint)

# In[32]:


c(data.lug_boot)

# In[33]:


c(data.safety)

# In[34]:


c(data.outcome)

# In[35]:


c(data.persons)

# In[36]:


c(data.doors)

# In[37]:


data.doors.replace('more', '5', inplace=True)

# In[11]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

# In[25]:


data.buying = enc.fit_transform(data.buying)
data.maint = enc.fit_transform(data.maint)
data.lug_boot = enc.fit_transform(data.lug_boot)
data.safety = enc.fit_transform(data.safety)

# In[26]:


data.head()

# In[27]:


X = data.loc[:, :'safety']
y = data.outcome

# In[28]:


X.head()

# In[29]:


from sklearn.model_selection import train_test_split

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

# In[31]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

# In[32]:


model.fit(X_train, y_train)

# In[33]:


y_pred = model.predict(X_test)

# In[36]:


print(accuracy_score(y_test, y_pred))
pd.crosstab(y_test, y_pred)

# In[35]:


# In[ ]:


# !pip install imblearn


# In[45]:


from imblearn.over_sampling import SMOTE

smote_res = SMOTE()
X_train_res, y_train_res = smote_res.fit_sample(X_train, y_train, random_state=10)

# In[ ]:


Counter(y_train_res)

# In[ ]:


Counter(y_train)

# In[46]:


model1 = KNeighborsClassifier()
model1.fit(X_train_res, y_train_res)

# In[47]:


y_pred_res = model1.predict(X_test)

# In[44]:


pd.crosstab(y_test, y_pred_res)

# In[48]:


pd.crosstab(y_test, y_pred)
