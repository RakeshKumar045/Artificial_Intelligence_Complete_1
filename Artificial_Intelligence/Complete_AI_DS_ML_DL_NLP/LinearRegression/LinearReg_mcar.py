# coding: utf-8

# In[25]:


import pandas as pd

# In[26]:


# get the folder path
# %pwd


# In[27]:


cars = pd.read_csv('mtcars.csv')
print(cars.head(2))

# In[28]:


# cars.shape


# In[29]:


temp = []
temp = cars.iloc[0, :]

# temp

# In[30]:


print(temp)

# In[31]:


print(cars.head(2))

# In[32]:


cars.iloc[0, :] = cars.iloc[1, :]
cars.iloc[1, :] = temp

# In[33]:


print(cars.head(2))

# In[34]:


print(cars.shape)

# In[35]:


# Feature Engineering
X = cars.loc[:, ['hp', 'wt', 'am']]
y = cars.mpg

# In[36]:


# Train and Test_Amat split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# In[37]:


print(X_train.head(2))

# In[38]:


# Importing and Initiating Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# In[39]:


# Training model with data
model.fit(X_train, y_train)

# In[40]:


# Model Evaluation -- predicint y for X_test
y_predict = model.predict(X_test)

# In[41]:


print(y_predict)

# In[42]:


print(y_test)

# In[43]:


# Evaluation Model efficients with Test_Amat data
from sklearn.metrics import r2_score

print("Accuracy : ", r2_score(y_test, y_predict))
