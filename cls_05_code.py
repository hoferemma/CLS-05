import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# read csv files
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# building our vectors (count words in each tweet and turn into data)
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])

#build a linear model
clf = linear_model.RidgeClassifier()

#use cross validation to test our model 
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring="f1")

# predictions on training set
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)