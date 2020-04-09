# coding: utf-8

# #  Naive Bayes Classifiers

# In[29]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# ## Naive Bayes
# ### Using Naive Bayes to predict spam

# In[30]:


# Use Latin encoding as the Data has non UFT-8 Chars
data = pd.read_csv("abhilasha_dataset_updated.csv", encoding='latin-1')

# In[31]:


data.columns

# In[32]:


data.isnull().sum()

# In[33]:


data.dropna(subset=['category'], how='all', inplace=True)

# In[64]:


data.isnull().sum()

# In[35]:


data.shape

# In[65]:


data.head(1)

# In[66]:


X = data.description
y = data.category

# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# In[68]:


y_test.describe()

# In[69]:


# X_test


# In[70]:


# TfidfVectorizer() is better than CountVectorizer() 
# TfidfVectorizer() has more feature than CountVectorizer()
vectorizer = TfidfVectorizer()

# In[71]:


X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names()

# In[72]:


len(feature_names)

# In[44]:


X_train_transformed

# In[93]:


feature_names[0:5394]

# In[94]:


X_train_transformed.toarray()

# In[96]:


#### slim the data for training and testing
selector = SelectPercentile(percentile=99)  # i need only 20% of data total =7510 , total of 10% = 751
selector.fit(X_train_transformed, y_train)
X_train_transformed_per = selector.transform(X_train_transformed).toarray()
X_test_transformed_per = selector.transform(X_test_transformed).toarray()

# In[97]:


X_train_transformed_per.shape

# In[98]:


#  it is good for confusion matrix , accuracry = 0.9798994974874372
# Confusion matrix :
# array([[1200,    1],
#        [  27,  165]])
clf = BernoulliNB()

# In[99]:


# 
#  it is not good for confusion matrix , accuracry = 0.9676956209619526
# clf = GaussianNB()

# Confusion matrix :
# array([[1172,   29],
#       [  16,  176]])


# In[100]:


# it is not good for efficiency , accuracy = 0.9382627422828428
# Confusion matrix :
# array([[1201,    0],
#       [  86,  106]])

# clf = MultinomialNB()


# In[101]:


clf.fit(X_train_transformed_per, y_train)
y_predict = clf.predict(X_test_transformed_per)
print(accuracy_score(y_test, y_predict))

# In[102]:


confusion_matrix(y_test, y_predict)

# In[103]:


clf_mul = MultinomialNB()
clf_mul.fit(X_train_transformed_per, y_train)
y_predict_mul = clf_mul.predict(X_test_transformed_per)
print(accuracy_score(y_test, y_predict_mul))

# In[104]:


confusion_matrix(y_test, y_predict_mul)

# In[105]:


accuracy_score(y_test, y_predict_mul)

# In[106]:


clf_ber = BernoulliNB()
clf_ber.fit(X_train_transformed_per, y_train)
y_predict_ber = clf_ber.predict(X_test_transformed_per)
print(accuracy_score(y_test, y_predict_ber))

# In[107]:


accuracy_score(y_test, y_predict_ber)

# In[108]:


pd.crosstab(y_test, y_predict_ber)
