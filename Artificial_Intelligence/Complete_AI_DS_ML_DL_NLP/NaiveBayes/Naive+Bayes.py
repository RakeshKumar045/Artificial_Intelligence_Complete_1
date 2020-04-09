# coding: utf-8

# #  Naive Bayes Classifiers

# In[5]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# ## Naive Bayes
# ### Using Naive Bayes to predict spam

# In[6]:


dataset = pd.read_csv("spambase.data.csv")
dataset.head()
dataset.shape

# In[7]:


dataset.head()

# In[8]:


X = dataset.iloc[:, 0:48]
y = dataset.iloc[:, -1]

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=17)

# In[13]:


from sklearn.metrics import confusion_matrix

BernNB = BernoulliNB(binarize=True)
BernNB.fit(X_train, y_train)
print(BernNB)

y_pred = BernNB.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# In[15]:


MultiNB = MultinomialNB()

MultiNB.fit(X_train, y_train)
print(MultiNB)

y_pred = MultiNB.predict(X_test)
accuracy_score(y_test, y_pred)

# In[16]:


GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
print(GausNB)

y_pred = GausNB.predict(X_test)
accuracy_score(y_test, y_pred)

# In[18]:


BernNB = BernoulliNB()
BernNB.fit(X_train, y_train)
print(BernNB)

y_pred = BernNB.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

# P(Q|F) = P(F|Q)*P(Q)/P(F)

# In[ ]:


df = pd.read_csv("spam.csv", encoding='latin-1')

# In[ ]:


df.head()

# In[ ]:


data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1,
    test_size=0.1,
    random_state=42)

# In[ ]:


print(data_train[:10])

# In[ ]:


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

# In[ ]:


data_train_transformed = vectorizer.fit_transform(data_train)
data_test_transformed = vectorizer.transform(data_test)

# In[ ]:


print(data_train_transformed[:10])

# In[ ]:


# slim the data for training and testing
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(data_train_transformed, labels_train)
data_train_transformed = selector.transform(data_train_transformed).toarray()
data_test_transformed = selector.transform(data_test_transformed).toarray()

# In[ ]:


print(data_train_transformed[:10])

# In[ ]:


clf = GaussianNB()
clf.fit(data_train_transformed, labels_train)
predictions = clf.predict(data_test_transformed)

print(accuracy_score(labels_test, predictions))

# In[ ]:


NewEmail = pd.Series(["Hi there, For are premium phone services call 08718711108"], index=[8000])
NewEmail

# In[ ]:


NewEmail_transformed = vectorizer.transform(NewEmail)
NewEmail_transformed = selector.transform(NewEmail_transformed).toarray()
clf.predict(NewEmail_transformed)

# In[ ]:


clf2 = GaussianNB()
clf2.fit(data_train, labels_train)
predictions = clf.predict(data_test)

print(accuracy_score(labels_test, predictions))

# In[ ]:


labels_train[:20]

# In[ ]:


# assigning predictor and target variables
x = np.array([[-3, 7], [1, 5], [1, 2], [-2, 0], [2, 3], [-4, 0], [-1, 1], [1, 1], [-2, 2], [2, 7], [-4, 1], [-2, 7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# In[ ]:


model = GaussianNB()

# Train the model using the training sets 
model.fit(x, Y)

# Predict Output
predicted = model.predict([[1, 2], [2, 7]])
print(predicted)
