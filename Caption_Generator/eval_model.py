from numpy import argmax
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import load_data as ld

#map a integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, image, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen = max_length)
		# predict next word
		yhat = model.predict([image, sequence], verbose = 0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, images, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, images[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# load training dataset (6K)
filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
#filename = '/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Flickr8k_text/Flickr_8k.trainImages.txt'
train = ld.load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = ld.load_clean_descriptions('data/descriptions.txt', train)
#train_descriptions = load_clean_descriptions('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
#train_features = load_image_features('data/features.pkl', train)
#train_features = load_image_features('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/features.pkl', train)
#print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = ld.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = ld.max_length(train_descriptions)
print('Description Length: %d' % max_length)
pickle.dump(tokenizer, open('data/tokenizer.pkl', 'wb'))

'''
# load test set
filename = '../Flickr8k_text/Flickr_8k.devImages.txt'
#filename = '/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Flickr8k_text/Flickr_8k.testImages.txt'
test = ld.load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = ld.load_clean_descriptions('data/descriptions.txt', test)
#test_descriptions = load_clean_descriptions('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = ld.load_image_features('data/features.pkl', test)
#test_features = load_image_features('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'caption_model.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
'''