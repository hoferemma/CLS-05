"""
COMP 4720
Group: CLS-05
Members: Emma Hofer & Alicia Ruz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing



# read csv files
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# apply tf-idf vecotirzer on raw text
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
train_vectors = tfidf_vectorizer.fit_transform(train_df["text"])
test_vectors = tfidf_vectorizer.transform(test_df["text"])

#build a linear model
clf = linear_model.RidgeClassifier()

#use cross validation to test our model 
scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring="f1")
print("F1 CV score:", scores.mean())

# predictions on training set
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("CLS-05-CKPT2.csv", index=False)