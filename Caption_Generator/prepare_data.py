from os import listdir, path
from pickle import dump
import string
from keras.applications.vgg16 import VGG16, preprocess_input 
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

## prepare image data
# extract feature from images
def extract_features(directory):
	# load model
	model = VGG16()
	# restruct the model
	model.layers.pop()
	model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
	# summarize
	print(model.summary())
	# extract feature from each image
	features = {}
	for name in listdir(directory):
		# load an image from file
		filename = path.join(directory, name)
		img = load_img(filename, target_size = (224, 224))
		# convert the image pixels to a numpy array
		img_array = img_to_array(img)
		# reshape data for the model
		img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
		# prepare the image for the VGG model
		img_array = preprocess_input(img_array)
		# get features
		feature = model.predict(img_array, verbose = 0)
		# get image id
		img_id = name.split('.')[0]
		# store feature
		features[img_id] = feature
		print("[INFO] extract features from {0}/{1}".format(name, len(listdir(directory))))
	return features

## prepare text data
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping
 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# extract features from all images
directory = '../Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))

filename = '../Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions.txt')