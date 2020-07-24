#!/usr/bin/env python3

print('Load data')
import joblib
vectorizer = joblib.load('vectorizer.joblib') 
from datavectorizer import loadVectorize
sentenceIDs, contribsmatrix, causalgolds, vectorizer, sentences = loadVectorize('../data/subtask1_test.csv', vectorizer)

print('Predict labels')
import joblib
model = joblib.load('model.joblib') 
labels = []
labels.append(['sentenceID', 'pred_label'])
modellabels = model.predict(contribsmatrix)
for sentenceIDindex in range(len(sentenceIDs)):
	labels.append([sentenceIDs[sentenceIDindex], modellabels[sentenceIDindex]])

print('Write labels')
import csv
with open('subtask1.csv', 'w') as labelcsv:
	writer = csv.writer(labelcsv)
	for label in labels:
		writer.writerow(label)


print('Zip file')
import zipfile
with zipfile.ZipFile('../data/subtask1.zip', mode='w') as zf:
	zf.write('subtask1.csv')
	zf.close()
import os
#os.remove('subtask1.csv')
