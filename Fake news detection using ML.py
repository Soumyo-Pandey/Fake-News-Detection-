import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))
df = pd.read_csv('../input/data.csv')



#replacing Body nan with Headline
for i in range(0,df.shape[0]-1):
    if(df.Body.isnull()[i]):
        df.Body[i] = df.Headline[i]
        
y = df.Label
X = df.Body

#train_test separation
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

#Applying tfidf to the data set
tfidf_vect = TfidfVectorizer(stop_words = 'english')
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())

#Applying Naive Bayes
clf = MultinomialNB() 
clf.fit(tfidf_train, y_train)                       # Fit Naive Bayes classifier according to X, y
pred = clf.predict(tfidf_test)                     # Perform classification on an array of test vectors X.
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)

#Applying Passive Aggressive classifier
linear_clf = PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)
