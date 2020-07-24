#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def loadVectorize(file, vectorizer=None):
	print('Import data and compute features')
	import csv, numpy, flair, nltk
	glove_embedding = flair.embeddings.WordEmbeddings('glove')
	document_embeddings = flair.embeddings.DocumentPoolEmbeddings([glove_embedding], pooling='max')
	sentences = []
	sentenceIDs = []
	causalgolds = []
	causalfeatures = []
	with open(file) as datacsv:
		for datacsvline in csv.reader(datacsv, delimiter=',', quotechar='"'):
			if not vectorizer:
				sentenceID,gold_label,sentence = datacsvline
			else:
				sentenceID,sentence = datacsvline
			if not sentenceID == 'sentenceID':
				sentenceIDs.append(sentenceID)
				sentences.append(sentence)
				if not len(sentences)%100:
					print('-', len(sentences), 'sentences')
				sentcausalfeatures = []
				sentcausalfeatures.append(len(sentence))
				sentcausalfeatures.append(len([c for c in sentence if c in ',;:.?!']))
				flairSentence = flair.data.Sentence(sentence)
				document_embeddings.embed(flairSentence)
				sentcausalfeatures.append(flairSentence.get_embedding())
				sentence_toks = nltk.word_tokenize(sentence)
				sentence_toks_len = len(sentence_toks)
				sentcausalfeatures.append(sentence_toks_len)
				sentence_tags = nltk.pos_tag(sentence_toks)
				sentence_tags_list = [t[1] for t in sentence_tags]
				for tag in ['IN', 'MD', 'VB', 'VBD', 'VBN', 'VBP']:
					nbtag = len([toktag for toktag in sentence_tags if toktag[1]==tag])
					sentcausalfeatures.append(nbtag)
					sentcausalfeatures.append(nbtag/sentence_toks_len)
					tagindex = 100
					if tag in sentence_tags_list:
						tagindex = sentence_tags_list.index(tag)
					sentcausalfeatures.append(tagindex)
					sentcausalfeatures.append(tagindex/sentence_toks_len)
				causalfeatures.append(sentcausalfeatures)
				if not vectorizer:
					causalgolds.append(int(gold_label))
	print('Vectorize texts')
	import numpy, scipy
	if vectorizer:
		contribsmatrix = vectorizer.transform(sentences)
	else:
		from sklearn.feature_extraction.text import CountVectorizer
		vectorizer = CountVectorizer(lowercase=False, ngram_range=(1, 3),)
		contribsmatrix = vectorizer.fit_transform(sentences)
	for ifeat in range(len(causalfeatures[0])):
		causalfeature = numpy.vstack([c[ifeat] for c in causalfeatures])
		if len(causalfeature.shape) == 1:
			causalfeature.reshape(-1,1)
		contribsmatrix = scipy.sparse.hstack([contribsmatrix, causalfeature])
	return sentenceIDs, contribsmatrix, causalgolds, vectorizer, sentences
