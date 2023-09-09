import numpy as np
import pandas as pd

import json
import csv

# dataset = pd.read_csv('toPredict.csv')
dataset = pd.read_json('toPredict.json')

# DATA CLEANING

import re
import nltk

# nltk.download('stopwords')      #Only once

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus=[]

for i in range(0, dataset.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

# DATA TRANSFORMATION
# Loading BoW dictionary

from sklearn.feature_extraction.text import CountVectorizer
import pickle
cvFile='BoW_SA_Model_HR.pkl'
cv = pickle.load(open(cvFile, "rb"))

X_fresh = cv.transform(corpus).toarray()

# PREDICTIONS
import joblib
classifier = joblib.load('Classifier_SA_Model_HR')

y_pred = classifier.predict(X_fresh)

predictions = pd.DataFrame(y_pred)
dataset = dataset.join(predictions)
# dataset['predicted_label'] = y_pred.tolist()

# dataset.to_csv("Predicted.csv", sep='\t', encoding='UTF-8', index=False)
dataset.to_json("Predicted.json")