# this example performs binary classification of movie reviews (good or bad), based on IMDB dataset

import tensorflow as tf
from tensorflow import keras

import numpy as np

print("TensorFlow version: ", tf.__version__)

# grab data set
imdb = keras.datasets.imdb

# load the training and test data, num_words=10000 means keep the 10000 most common words in data
(train_data, train_labels), (test_data,
 test_labels) = imdb.load_data(num_words=10000)

# get a sense of the size of the data set
print("Training data size: {}, labels: {}".format(
    len(train_data), len(train_labels)))
print("Number of words in first: {}, and second review: {}".format(
    len(train_data[0]), len(train_data[1])))

# the reviews are stored as integer vectors. each integer corresponds to one word
print("training_data[0]=")
print(train_data[0])

# we need to convert back to words, at least to get human readable reviews, so grab the dictionary
word_index = imdb.get_word_index()

# The first few indices [0,1,2,3] are reserved and won't map to words
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown. probably a word not in our top 10000 words
word_index["<UNUSED>"] = 3

# it's useful to have the reverse mapping as well
reverse_word_index = {value: key for key, value in word_index.items()}

# helper function for decoding the vector reviews back into text


def decode_review(review):
  return " ".join([reverse_word_index.get(i, "<?>") for i in review])


# now lets see the first review again
print("decoded training_data[0]=")
print(decode_review(train_data[0]))

'''
The reviews—the arrays of integers—must be converted to tensors before fed into the neural network.
This conversion can be done a couple of ways:

Convert the arrays into vectors of 0s and 1s indicating word occurrence, similar to a one-hot 
encoding. For example, the sequence [3, 5] would become a 10,000-dimensional vector that is all
zeros except for indices 3 and 5, which are ones. Then, make this the first layer in our network—a
Dense layer—that can handle floating point vector data. This approach is memory intensive, though,
requiring a num_words * num_reviews size matrix.

Alternatively, we can pad the arrays so they all have the same length, then create an integer tensor
of shape max_length * num_reviews. We can use an embedding layer capable of handling this shape as
the first layer in our network.
'''

# movie reviews must be the same length, we'll use the pad_sequences function to standardize lengths
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# data now all padded at the end with zeros

## now lets decide on a model architecture
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) # layer that learns word embeddings, 16 dimensional space
model.add(keras.layers.GlobalAveragePooling1D()) # output of embedding layer is 2D, this flattens it by averging over sequence dimension
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) # binary sigmoid output of 0 or 1

# look at the structure
model.summary()

model.compile(optimizer='adam',
              loss ='binary_crossentropy', # good for binary probabilities
              metrics=['acc']) # accuracy

# for cross entropy (and in general) we want a validation set, so pull some aside from train_data
validation_data = train_data[:10000]
validation_labels = train_labels[:10000]

remaining_train_data = train_data[10000:]
remaining_train_labels = train_labels[10000:]

# now train model on data while monitoring performance on validation set
history = model.fit(remaining_train_data,
                    remaining_train_labels,
                    epochs=40,
                    batch_size=512,
                    validation_data =(validation_data, validation_labels),
                    verbose=True)

# let's evaluate the performance of our model
results = model.evaluate(test_data, test_labels)
print('model results on test data: {}'.format(results)) # about 87% accurate

# we can plot our training and validation accuracy as a function of training
history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history_dict['acc'] # training accuracy
val_acc = history_dict['val_acc'] # validation accuracy
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title("Training loss and validation loss scores")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(epochs, acc, 'go', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title("Training and validation accuracy scores")
plt.xlabel('Epochs')
plt.ylabel('Accuracy [%]')
plt.legend()

plt.show()




