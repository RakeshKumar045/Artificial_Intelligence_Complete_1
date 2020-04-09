# coding: utf-8

# In[10]:


from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# In[16]:


twenty_train.target_names

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

# In[12]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# In[14]:


from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# In[15]:


from sklearn.metrics import accuracy_score

twenty_test = fetch_20newsgroups(subset='tensorbroad_pb_android', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
accuracy_score(predicted, twenty_test.target)

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }
import time

start = time.time()
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
print(gs_clf.best_score_, gs_clf.best_params_)
end = time.time()
print(end - start)

# In[ ]:


import time
from sklearn.model_selection import GridSearchCV

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (0.01, 0.001),
                  }
start = time.time()
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)
print(gs_clf.best_score_, gs_clf.best_params_)
end = time.time()
print(end - start)

# ## Random Search  Saves Computation and time.

# In[ ]:


from sklearn.model_selection import

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }
import time

start = time.time()
gs_clf = RandomizedSearchCV(text_clf, parameters, n_jobs=-1, n_iter=1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
print(gs_clf.best_score_, gs_clf.best_params_)
end = time.time()
print(end - start)

# ## Removing stop words:
# (the, then etc) from the data. You should do this only when stop words are not useful for the underlying problem. In most of the text classification problems, this is indeed not useful. Let’s see if removing stop words increases the accuracy. 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

# In[ ]:


from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='tensorbroad_pb_android', shuffle=True)
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)
accuracy_score(predicted, twenty_test.target)

# ## Stemming

# In[ ]:


import nltk

nltk.download()

# In[ ]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                             ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False)),
                             ])
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

# In[ ]:


import numpy as np

np.mean( == )


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(predicted_mnb_stemmed, twenty_test.target)
