# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# In[2]:


pass_data = pd.read_csv('pass.csv')

# In[3]:


pass_data.head()

# In[4]:


pass_data.shape

# In[5]:


from sklearn.preprocessing import StandardScaler

X = pass_data.loc[:, ['Hours']]
y = pass_data.Pass

# In[6]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X[1:5]

# In[7]:


X[1:5]

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=1)

# In[9]:


model_logr = LogisticRegression()

# In[10]:


model_logr.fit(X_train, y_train)

# In[11]:


y_predict = model_logr.predict(X_test)

# In[12]:


y_predict

# In[13]:


y_test

# In[14]:


accuracy_score(y_predict, y_test)

# In[15]:


confusion_matrix(y_predict, y_test)


# In[16]:


def student_result(hours):
    if (model_logr.predict(hours) == 1):
        return "PASS"
    else:
        return "FAIL"


# In[17]:


student_result(0.5)

# In[18]:


import numpy as np

# In[19]:


X = np.array(pass_data.Hours)
y = np.array(pass_data.Pass)

plt.scatter(X, y)

# In[ ]:


import numpy as np

X_generated = np.linspace(0, 10, 50)
# y_predicted = student_result()

y_predicted = []
for x in X_generated:
    y_predicted.append(int(model_logr.predict(x)))

# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(X_generated, np.array(y_predicted))
# plt.scatter(X,y,c='green',s=100)


# In[ ]:


plt.scatter(X, y, c='green', s=100)

# In[20]:


h = np.array([10, 2, 4, 8])

# In[21]:


h.std()
