# coding: utf-8

# #  Naive Bayes Classifiers

# In[24]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# ## Naive Bayes
# ### Using Naive Bayes to predict spam

# In[25]:


# Use Latin encoding as the Data has non UFT-8 Chars
data = pd.read_csv("spam.csv", encoding='latin-1')

# In[26]:


data.columns

# In[27]:


# rename the columnns
data.columns = ['v1', 'v2', 'Rename_Unnamed: 2', 'Rename_Unnamed: 3', 'Unnamed: 4']

# In[28]:


data.shape

# In[29]:


# data.v2


# In[30]:


data.head(5)

# In[31]:


data[data.v1 == 'spam'].count()

# In[32]:


# data.v1 # it is spam or not
# data.v2 # it is text


# In[33]:


X = data.v2
y = data.v1

# In[34]:


X.head()

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# In[36]:


X_test.head()

# In[37]:


y_test.describe()

# In[38]:


# TfidfVectorizer() is better than CountVectorizer() 
# TfidfVectorizer() has more feature than CountVectorizer()
vectorizer = TfidfVectorizer()

# In[39]:


X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names()

# In[40]:


len(feature_names)

# In[41]:


feature_names

# In[42]:


feature_names[2000:2005]

# In[43]:


X_train_transformed.toarray()

# In[44]:


#### slim the data for training and testing
selector = SelectPercentile(percentile=10)  # i need only 20% of data total =7510 , total of 10% = 751
selector.fit(X_train_transformed, y_train)
X_train_transformed_per = selector.transform(X_train_transformed).toarray()
X_test_transformed_per = selector.transform(X_test_transformed).toarray()

# In[45]:


X_train_transformed_per.shape

# In[61]:


X_test_transformed_per

# In[62]:


X_test_transformed

# In[46]:


#  it is good for confusion matrix , accuracry = 0.9798994974874372
# Confusion matrix :
# array([[1200,    1],
#        [  27,  165]])
clf = BernoulliNB()

# In[47]:


# 
#  it is not good for confusion matrix , accuracry = 0.9676956209619526
# clf = GaussianNB()

# Confusion matrix :
# array([[1172,   29],
#       [  16,  176]])


# In[48]:


# it is not good for efficiency , accuracy = 0.9382627422828428
# Confusion matrix :
# array([[1201,    0],
#       [  86,  106]])

# clf = MultinomialNB()


# In[49]:


clf.fit(X_train_transformed_per, y_train)
y_predict = clf.predict(X_test_transformed_per)

# In[50]:


print(accuracy_score(y_test, y_predict))

# In[51]:


confusion_matrix(y_test, y_predict)

# In[52]:


NewEmail = pd.Series(["HI.. we have meeting today.. please attend "])
NewEmail

# In[53]:


NewEmail_transformed = vectorizer.transform(NewEmail)
NewEmail_transformed = selector.transform(NewEmail_transformed).toarray()
clf.predict(NewEmail_transformed)

# In[54]:


clf_mul = MultinomialNB()
clf_mul.fit(X_train_transformed_per, y_train)
y_predict_mul = clf_mul.predict(X_test_transformed_per)

# In[55]:


confusion_matrix(y_test, y_predict_mul)

# In[56]:


accuracy_score(y_test, y_predict_mul)

# In[57]:


clf_ber = BernoulliNB()
clf_ber.fit(X_train_transformed_per, y_train)
y_predict_ber = clf_ber.predict(X_test_transformed_per)

# In[58]:


accuracy_score(y_test, y_predict_ber)

# In[59]:


confusion_matrix(y_test, y_predict_ber)

# In[60]:


pd.crosstab(y_test, y_predict_ber)
