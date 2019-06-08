import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from generate_model import BahdanauAttention, CNN_Encoder, RNN_Decoder
from generate_model import loss_function

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512

# Load the numpy files
def map_func(img_name, cap):
    #img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    img_tensor = np.load(img_name.decode('utf-8').replace("mscoco2014", "features_incepv3").replace(".jpg", ".npy"))
    return img_tensor, cap

def data_split(img_name_vector, train_captions, tokenizer):
    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                            cap_vector,
                                                                            test_size=0.2,
                                                                            random_state=0)
    # Save img_name_val, cap_val to disk for testing
    #pickle.dump(img_name_val, open('/content/drive/My Drive/XLA_UD/models/img_names_test.pkl', 'wb'))
    #pickle.dump(cap_val, open('/content/drive/My Drive/XLA_UD/models/captions_test.pkl', 'wb'))
    pickle.dump(img_name_val, open('../models/img_names_test.pkl', 'wb'))
    pickle.dump(cap_val, open('../models/captions_test.pkl', 'wb'))

    return img_name_train, img_name_val, cap_train, cap_val

def data_generator(img_name_train, cap_train):
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    #dataset = tf.data.Dataset.from_tensor_slices((img_name_vector, cap_vector))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

@tf.function
def train_step(decoder, encoder, optimizer, tokenizer, img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

def train_model(EPOCHS = 20):
    # load captions and images for training
    train_captions = pickle.load(open('../models/train_captions.pkl', 'rb'))
    img_name_vector = pickle.load(open('../models/img_name_vector.pkl', 'rb'))
    tokenizer = pickle.load(open('../models/tokenizer.pkl', 'rb'))

    img_name_train, img_name_val, cap_train, cap_val = data_split(img_name_vector, train_captions, tokenizer)

    vocab_size = len(tokenizer.word_index) + 1
    num_steps = len(img_name_train) // BATCH_SIZE

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "../models/checkpoints"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    
    loss_plot = []

    dataset = data_generator(img_name_train, cap_train)

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(decoder, encoder, optimizer, tokenizer, img_tensor, target)
            total_loss += t_loss

            if batch % 200 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

if __name__ == "__main__":
    train_model(EPOCHS=31)