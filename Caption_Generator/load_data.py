import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	# read all text
	text = file.read()
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = []
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def load_clean_descriptions(filename, dataset):
	doc = load_doc(filename)
	descriptions = {}
	for line in doc.split('\n'):
		tokens = line.split()
		img_id, img_desc = tokens[0], tokens[1:]
		if img_id in dataset:
			if img_id not in descriptions:
				descriptions[img_id] = []
			# add the start and end of the sequence to description
			desc = 'startseq ' + ' '.join(img_desc) + ' endseq'
			descriptions[img_id].append(desc)

	return descriptions

# load image features
def load_image_features(filename, dataset):
	all_features = pickle.load(open(filename, 'rb'))
	# get features in dataset
	features = {k : all_features[k] for k in dataset}
	return features

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, desc_list, image):
	vocab_size = len(tokenizer.word_index) + 1
	X1, X2, y = list(), list(), list()
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X, y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output to one-hot-encoded vector
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			X1.append(image)
			X2.append(in_seq)
			y.append(out_seq)

	return np.array(X1), np.array(X2), np.array(y)

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, images, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			img = images[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, img)
			yield [[in_img, in_seq], out_word]