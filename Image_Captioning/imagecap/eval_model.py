import numpy as np
import tensorflow as tf
import pickle
import re
import json
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
    
    train_captions = pickle.load(open('../models/train_captions.pkl', 'rb'))
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    #attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    img_tensor_val = np.load(image.replace("mscoco2014", "features_incepv3").replace(".jpg", ".npy"))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        result.append(tokenizer.index_word[predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

# evaluate the skill of the model
def evaluate_model(encoder, decoder, tokenizer):
    img_names_test = pickle.load(open('../models/img_names_test.pkl', 'rb'))
    #captions_test = pickle.load(open('../models/captions_test.pkl', 'rb'))

    name_check = re.compile("COCO_train2014_0*")

    print("img_names_test: ", len(img_names_test))
    img_names_test = sorted(set(img_names_test))
    print("unique_img_names_test: ", len(img_names_test))
    
    # Read the json file
    annotation_file = '../annotations/captions_train2014.json'
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    actual, predicted = list(), list()
    count = 0
    for image_name in img_names_test:
        image_id = image_name.split("/")[-1]
        image_id = name_check.sub("",image_id).replace(".jpg", "")
        
        # generate description
        count += 1
        print("Image_id: {0} / {1}".format(image_id, count))
        yhat = generate_desc(image_name, tokenizer, encoder, decoder)
        # generate references
        ref = list()
        for annot in annotations['annotations']:
            if int(image_id) == annot['image_id']:
                ref.append(annot["caption"].replace(".", "").split(" "))
            
        actual.append(ref)
        predicted.append(yhat)

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == "__main__":
    tokenizer = pickle.load(open('../models/tokenizer.pkl', 'rb'))
    
    vocab_size = len(tokenizer.word_index) + 1
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt.restore("../models/checkpoints/ckpt-6")

    evaluate_model(encoder, decoder, tokenizer)
