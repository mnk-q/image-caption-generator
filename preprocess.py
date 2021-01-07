import string
from utils import *

def toVocabulary(descriptions):
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

def saveDescriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = loadFile(filename)
descriptions = loadDescriptions(doc)
print('Loaded: %d ' % len(descriptions))
cleanDescriptions(descriptions)
vocabulary = toVocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
saveDescriptions(descriptions, 'descriptions.txt')