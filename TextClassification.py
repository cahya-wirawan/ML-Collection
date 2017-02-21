#!/opt/local/bin/python2.7

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

import numpy as np

random_state = 42

categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

""" Retrieve train newsgroup dataset """
twenty_train = fetch_20newsgroups(subset='train',
                                  # remove=('headers', 'footers', 'quotes'),
                                  categories=categories,
                                  shuffle=True,
                                  random_state=random_state)

print("Following newsgroups will be tested: {}".format(twenty_train.target_names))
print("Number of sentences: {}".format(len(twenty_train.data)))

""" Retrieve test newsgroup dataset """
twenty_test = fetch_20newsgroups(subset='test',
                                 # remove=('headers', 'footers', 'quotes'),
                                 categories=categories,
                                 shuffle=True,
                                 random_state=random_state)
docs_test = twenty_test.data

""" Naive Bayes classifier """
bayes_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())
                      ])
bayes_clf.fit(twenty_train.data, twenty_train.target)
""" Predict the test dataset using Naive Bayes"""
predicted = bayes_clf.predict(docs_test)
print('Naive Bayes correct prediction: {:4.2f}'.format(np.mean(predicted == twenty_test.target)))
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

""" Support Vector Machine (SVM) classifier"""
svm_clf = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=   5, random_state=42)),
])
svm_clf.fit(twenty_train.data, twenty_train.target)
""" Predict the test dataset using Naive Bayes"""
predicted = svm_clf.predict(docs_test)
print('SVM correct prediction: {:4.2f}'.format(np.mean(predicted == twenty_test.target)))
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))
