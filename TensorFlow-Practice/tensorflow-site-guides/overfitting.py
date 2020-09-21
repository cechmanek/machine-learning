# Using TensorFlow and Keras on the IMDB data set to explore overfitting and 
# underfitting

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print("tensorflow version: ", tf.__version__)

# we'll use the IMDB movie review with multi-hot encoding instead of word 
# embedding as before
NUM_WORDS = 10000
(train_data_, train_labels), (test_data_, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

train_data = np.zeros((len(train_data_), NUM_WORDS))

for i, word_index in enumerate(train_data_):
    train_data[i, word_index] = 1.0

test_data = np.zeros((len(test_data_), NUM_WORDS))
for i, word_index in enumerate(test_data_):
    test_data[i, word_index] = 1.0

# look at one of the training data vectors
plt.plot(train_data[0])
plt.show()

# let's heavily overfit using dense layers to see what happens
baseline_model = keras.Sequential([
                                  keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
                                  keras.layers.Dense(16, activation=tf.nn.relu),
                                  keras.layers.Dense(1, activation=tf.nn.sigmoid) # output layer
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics= ['accuracy', 'binary_crossentropy'])

# look at the model summary
print(baseline_model.summary())

# do a first pass at training
baseline_history = baseline_model.fit(train_data, train_labels, 
                                    epochs=20, batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# the training loss and accuracy keep improving, but validation scores get worse after only a few epochs
# this is the trademark sign of overfitting

# now make a smaller model that has fewer parameters to overfit with
smaller_model = keras.Sequential([
                                  keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
                                  keras.layers.Dense(4, activation=tf.nn.relu),
                                  keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(train_data, train_labels,
                                    epochs=20, batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# we can also create a huge model that is certain to overfit
bigger_model = keras.Sequential([
                                keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
                                keras.layers.Dense(512, activation=tf.nn.relu),
                                keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

# plotting the model histories helps to see what is happening
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Validation')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

plt.show()

