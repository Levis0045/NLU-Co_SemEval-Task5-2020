#!/usr/bin/env python3

print('-- Load data')
from datavectorizer import loadVectorize
sentenceIDs, contribsmatrix, causalgolds, vectorizer, sentences = loadVectorize('../data/task1-train_causal.csv')

print('-- Split train test')
from sklearn.model_selection import train_test_split
contribs_train, contribs_test, causal_train, causal_test = train_test_split(contribsmatrix, causalgolds, test_size=0.2, random_state=42)

print('-- Choose model')
from sklearn.svm import LinearSVC
model = LinearSVC(max_iter=1000000, class_weight={0:0.008})
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(random_state=0, max_iter=10000)

print('-- Learn model on train')
weights = [1+x/2 for x in causal_train]
model.fit(contribs_train, causal_train) #, sample_weight=weights)

print('-- Scores on test')
from sklearn.metrics import classification_report
report = classification_report(
	causal_test,
	model.predict(contribs_test),
	target_names=['no counter', 'counter']
)
print(report)

print('-- Learn model on train')
model.fit(contribsmatrix, causalgolds)

print('-- Save (vectorizer and model)')
import joblib
joblib.dump(vectorizer, 'vectorizer.joblib') 
joblib.dump(model, 'model.joblib') 
