from keras.preprocessing.text import Tokenizer
from pickle import dump
from utils import *

def createTokenizer(descriptions):
	lines = toLines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = loadFileSet(filename)
print('Dataset: %d' % len(train))
train_descriptions = loadCleanDescriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer = createTokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))