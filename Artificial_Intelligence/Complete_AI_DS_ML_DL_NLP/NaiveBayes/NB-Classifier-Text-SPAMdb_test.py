# coding: utf-8

# #  Naive Bayes Classifiers

# In[14]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# ## Naive Bayes
# ### Using Naive Bayes to predict spam

# In[2]:


# Use Latin encoding as the Data has non UFT-8 Chars
data = pd.read_csv("abhilasha_dataset_updated.csv", encoding='latin-1')

# In[3]:


data.columns

# In[4]:


data.isnull().sum()

# In[5]:


data.dropna(subset=['category'], how='all', inplace=True)

# In[6]:


data.isnull().sum()

# In[8]:


data.shape

# In[21]:


data.head(2)

# In[16]:


X = data.description
y = data.category

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# In[18]:


y_test.describe()

# In[22]:


# X_test


# In[23]:


# TfidfVectorizer() is better than CountVectorizer() 
# TfidfVectorizer() has more feature than CountVectorizer()
vectorizer = TfidfVectorizer()

# In[24]:


X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names()

# In[25]:


len(feature_names)

# In[16]:


feature_names[2000:2005]

# In[17]:


X_train_transformed.toarray()

# In[18]:


#### slim the data for training and testing
selector = SelectPercentile(percentile=10)  # i need only 20% of data total =7510 , total of 10% = 751
selector.fit(X_train_transformed, y_train)
X_train_transformed_per = selector.transform(X_train_transformed).toarray()
X_test_transformed_per = selector.transform(X_test_transformed).toarray()

# In[19]:


X_train_transformed_per.shape

# In[20]:


#  it is good for confusion matrix , accuracry = 0.9798994974874372
# Confusion matrix :
# array([[1200,    1],
#        [  27,  165]])
clf = BernoulliNB()

# In[21]:


# 
#  it is not good for confusion matrix , accuracry = 0.9676956209619526
# clf = GaussianNB()

# Confusion matrix :
# array([[1172,   29],
#       [  16,  176]])


# In[22]:


# it is not good for efficiency , accuracy = 0.9382627422828428
# Confusion matrix :
# array([[1201,    0],
#       [  86,  106]])

# clf = MultinomialNB()


# In[23]:


clf.fit(X_train_transformed_per, y_train)
y_predict = clf.predict(X_test_transformed_per)

# In[24]:


print(accuracy_score(y_test, y_predict))

# In[25]:


confusion_matrix(y_test, y_predict)

# In[26]:


NewEmail = pd.Series(["HI.. we have meeting today.. please attend "])
NewEmail

# In[27]:


NewEmail_transformed = vectorizer.transform(NewEmail)
NewEmail_transformed = selector.transform(NewEmail_transformed).toarray()
clf.predict(NewEmail_transformed)

# In[28]:


clf_mul = MultinomialNB()
clf_mul.fit(X_train_transformed_per, y_train)
y_predict_mul = clf_mul.predict(X_test_transformed_per)

# In[29]:


confusion_matrix(y_test, y_predict_mul)

# In[30]:


accuracy_score(y_test, y_predict_mul)

# In[31]:


clf_ber = BernoulliNB()
clf_ber.fit(X_train_transformed_per, y_train)
y_predict_ber = clf_ber.predict(X_test_transformed_per)

# In[32]:


accuracy_score(y_test, y_predict_ber)

# In[33]:


confusion_matrix(y_test, y_predict_ber)

# In[34]:


pd.crosstab(y_test, y_predict_ber)
