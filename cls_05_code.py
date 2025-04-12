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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# read csv files
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# apply tf-idf vecotirzer on raw text
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
train_vectors = tfidf_vectorizer.fit_transform(train_df["text"])
test_vectors = tfidf_vectorizer.transform(test_df["text"])

# logistic Regression setup (replacing linear model) 
logreg = LogisticRegression(max_iter=1000)

# parameter grid
param_grid = {
    "C": [0.01, 0.1, 0.5, 1, 5, 10]  #try multiple values
}

# use stratified 5-fold cross validation to test our model 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#perform grid search
grid = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=skf, scoring="f1", n_jobs=-1)
grid.fit(train_vectors, train_df["target"])

#best model and score
print("Best F1 CV Score:", grid.best_score_)
print("Best Parameters:", grid.best_params_)

#final model: predict on test set using the best logistic regression
best_model = grid.best_estimator_
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission["target"] = best_model.predict(test_vectors)

# Save predictions
sample_submission.to_csv("CLS-05-CKPT2.csv", index=False)