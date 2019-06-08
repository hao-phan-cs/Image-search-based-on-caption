import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import cv2
from prepare_data import load_image
from generate_model import BahdanauAttention, CNN_Encoder, RNN_Decoder

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

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

    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_captions = pickle.load(open('../models/train_captions.pkl', 'rb'))
    tokenizer = pickle.load(open('../models/tokenizer.pkl', 'rb'))

    vocab_size = len(tokenizer.word_index) + 1
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "../models/checkpoints"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

    '''
    ## captions on an image
    image_path = 'abc.jpg'
    result, attention_plot = generate_desc(image_path, tokenizer, encoder, decoder)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, attention_plot)
    # opening the image
    #Image.open(image_path)
    img = cv2.imread(image_path)
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    '''

    # captions on the test set
    img_names_test = pickle.load(open('../models/img_names_test.pkl', 'rb'))
    captions_test = pickle.load(open('../models/captions_test.pkl', 'rb'))

    rid = np.random.randint(0, len(img_names_test))
    image = img_names_test[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in captions_test[rid] if i not in [0]])
    result, attention_plot = generate_desc(image, tokenizer, encoder, decoder)

    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)
    # opening the image
    #Image.open(img_names_test[rid])
    img = cv2.imread(image)
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 