#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:07:35 2019

@author: rakeshkumar
"""

# import nltk import word_tokenize
from nltk.corpus import stopwords

# nltk.download()

word = stopwords.words("english")
# print(word)

word_test = stopwords.words('english')
print("hi ", word_test)

EXAMPLE_TEXT = “Hello
Mr.Nitin, what
are
you
doing
today? The
weather is dull, and NLTK is awesome.The
sky is pinkish - blue.You
shouldn’t
eat
cardboard.”

# print(sent_tokenize(EXAMPLE_TEXT))

# print(len(sent_tokenize(EXAMPLE_TEXT)))
