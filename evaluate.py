from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from utils import *


def createTokenizer(descriptions):
	lines = toLines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def maxLength(descriptions):
	lines = toLines(descriptions)
	return max(len(d.split()) for d in lines)

def wordForId(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generateDesc(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = wordForId(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

def evaluateModel(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	for key, desc_list in descriptions.items():
		yhat = generateDesc(model, tokenizer, photos[key], max_length)
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = loadFileSet(filename)
print('Dataset: %d' % len(train))
train_descriptions = loadCleanDescriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer = createTokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = maxLength(train_descriptions)
print('Description Length: %d' % max_length)
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = loadFileSet(filename)
print('Dataset: %d' % len(test))
test_descriptions = loadCleanDescriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
filename = 'model-ep002-loss3.245-val_loss3.612.h5'
model = load_model(filename)
evaluateModel(model, test_descriptions, test_features, tokenizer, max_length)