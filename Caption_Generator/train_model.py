from load_data import load_set, max_length, load_image_features
from load_data import load_clean_descriptions, create_sequences
from load_data import create_tokenizer, data_generator
from define_model import define_model
from keras.callbacks import ModelCheckpoint

# load training dataset (6K)
#filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
filename = '/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
#train_descriptions = load_clean_descriptions('data/descriptions.txt', train)
train_descriptions = load_clean_descriptions('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
#train_features = load_image_features('data/features.pkl', train)
train_features = load_image_features('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# load test set
#filename = '../Flickr8k_text/Flickr_8k.devImages.txt'
filename = '/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
#test_descriptions = load_clean_descriptions('data/descriptions.txt', test)
test_descriptions = load_clean_descriptions('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
#test_features = load_image_features('features.pkl', test)
test_features = load_image_features('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/data/features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# fit model 
# define the model
model = define_model(vocab_size, max_length)

'''
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
'''

# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# save model
model.save('/content/drive/My Drive/Me/Computer Vision/XLA_UD/Image-search-based-on-caption/Caption_Generator/caption_model.h5')