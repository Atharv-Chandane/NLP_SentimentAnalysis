from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import pandas as pd

dataset = pd.read_csv('Hotel_Reviews_ModelTrainer.csv')

# DATA PRE-PROCESSING


# nltk.download('stopwords')  #only while running the code first time

ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus = []

# 20491
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# DATA TRANSFORMATION

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Saving Bag of Words dictionary
bow_path = 'BoW_SA_Model_HR.pkl'
pickle.dump(cv, open(bow_path, "wb"))

# DIVIDING DATASET INTO TRAINING AND TEST SET

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# MODEL FITTING

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Exporting NaiveBayes Clissifier for future use in prediction
joblib.dump(classifier, 'Classifier_SA_Model_HR')
# MODEL PERFORMANCE
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print(acc)
