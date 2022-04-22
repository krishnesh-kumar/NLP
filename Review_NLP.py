# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:31 2022

@author: krish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
data=pd.read_csv("C:/Users/krish/OneDrive/Desktop/AI/NLP/Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
import re
corpus=[]
for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1300)
X=cv.fit_transform(corpus).toarray()
Y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10,random_state=2)
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

from sklearn.metrics import accuracy_score

score=accuracy_score(Y_test,Y_pred)
print(score)
