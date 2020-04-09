# coding: utf-8

# In[1]:


import pandas as pd
from collections import Counter as c
from imblearn.over_sampling import SMOTE

# In[11]:


data = pd.read_csv('bank-full.csv', ';')

# In[12]:


data.isnull().sum()

# In[13]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
enc.fit(data.job)
data.job = enc.transform(data.job)
data.head()

# In[14]:


enc.fit(data.default)
data.default = enc.transform(data.default)
data.head()

# In[15]:


enc.fit(data.housing)
data.housing = enc.transform(data.housing)
data.head()

# In[16]:


enc.fit(data.loan)
data.loan = enc.transform(data.loan)
data.head()

# In[17]:


enc.fit(data.marital)
data.marital = enc.transform(data.marital)
data.head()

# In[18]:


enc.fit(data.education)
data.education = enc.transform(data.education)

# In[19]:


X = data.loc[:, ['age', 'job', 'education', 'default', 'housing', 'loan', 'balance']]
y = data.y

# In[20]:


from sklearn.model_selection import train_test_split

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=10)

# In[22]:


from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=8)

# In[23]:


model_knn.fit(X_train, y_train)

# In[24]:


y_predict = model_knn.predict(X_test)

# In[25]:


from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test, y_predict)

# In[26]:


confusion_matrix(y_test, y_predict)

# In[32]:


c(y_test)

# In[28]:


from collections import Counter

# In[29]:


Counter(data.y)

# In[30]:


# In[31]:


smote_res = SMOTE()
x_train_res, y_train_res = smote_res.fit_sample(X_train, y_train)
c(y_train_res)
