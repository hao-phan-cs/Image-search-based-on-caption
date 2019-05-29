# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
from nltk.translate.bleu_score import corpus_bleu
from prepare_data import load_image
from generate_model import BahdanauAttention, CNN_Encoder, RNN_Decoder

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
#attention_features_shape = 64

embedding_dim = 256
units = 512

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def generate_desc(image, tokenizer, encoder, decoder):
    # load InceptionV3
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    #----------------------------------------------------------------------
    train_captions = pickle.load(open('/content/drive/My Drive/XLA_UD/models/train_captions.pkl', 'rb'))
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)
    #----------------------------------------------------------------------

    #attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result

# evaluate the skill of the model
def evaluate_model(encoder, decoder, descriptions, photos, tokenizer):
    img_names_test = pickle.load(open('/content/drive/My Drive/XLA_UD/models/img_names_test.pkl', 'rb'))
    captions_test = pickle.load(open('/content/drive/My Drive/XLA_UD/models/captions_test.pkl', 'rb'))

    actual, predicted = list(), list()
	# step over the whole set
    for image, caption_vec in zip(img_names_test, captions_test):
        # generate description
        yhat = generate_desc(image, tokenizer, encoder, decoder)
        # store actual and predicted
        real_caption = [tokenizer.index_word[i] for i in caption_vec if i not in [0]]
        actual.append(real_caption)
        predicted.append(yhat)

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

'''
#--------------------------------------------main()-------------------------------------------------
train_captions = pickle.load(open('/content/drive/My Drive/XLA_UD/models/train_captions.pkl', 'rb'))
tokenizer = pickle.load(open('/content/drive/My Drive/XLA_UD/models/tokenizer.pkl', 'rb'))

vocab_size = len(tokenizer.word_index) + 1 + 1 # bug here: vocab_size = 8235 (real 8236)
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()

checkpoint_path = "/content/drive/My Drive/XLA_UD/checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

image_path = '/content/drive/My Drive/XLA_UD/COCO_test2015_000000000108.jpg'
result = evaluate(image_path, tokenizer, encoder, decoder)
print ('Prediction Caption:', ' '.join(result))
# opening the image
Image.open(image_path)
'''