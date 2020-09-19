'''
multi-class classification if news stories
following the code starting on page 78
slightly modified to use tensorflows native keras

Unlike the imdb.py example, this model works just with, even at batch_size=512
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print('the size of our training set is ', len(train_data))
print('the size of our test set is ', len(test_data))

# the data comes in numeric form (words transformed to indices in a dictionary)
# grab the dictionary so we can translate

word_index = reuters.get_word_index()
reverse_word_index = {val: key for key, val in word_index.items()}

decoded_first_sample = [reverse_word_index.get(i-3, '?') for i in train_data[0]]
# indices 0,1,2 are reserved for 'padding', 'sequence start', and 'unknown'

# need to vectorize the training data from [1,4,1,4,6,8,4,3,7,35]
# to [0,1,0,0,0,0,0...] #1
#    [0,0,0,0,1,0,0...] #4
# basically one-hot encoding the sample
def vectorize_sequences(sequences, dimensions=10000):
    result = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1.0

    return result

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# we need to one-hot encode the multi-class labels
# manually, this looks like
def one_hot(labels, dimensions=46):
    results = np.zeros((len(labels),dimensions))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results

y_train = one_hot(train_labels)
y_test = one_hot(test_labels)

# keras has this method, so we'll use it instead
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# time to build a model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax')) # 46 possible classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# time to train
history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=20, batch_size=512)

# time to plot results
history_dict = history.history
print(history_dict.keys())
training_loss = history_dict['loss']
training_accuracy = history_dict['accuracy']
validation_loss = history_dict['val_loss']
validation_accuracy = history_dict['val_accuracy']

epochs = range(1,len(training_loss)+1) # == 20

plt.plot(epochs, training_loss, 'bo', label='training loss')
plt.plot(epochs, training_accuracy, 'b', label='training accuracy')
plt.plot(epochs, validation_loss, 'ro', label='validation loss')
plt.plot(epochs, validation_accuracy, 'r', label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend() # legend grabs the labels specified above

plt.show()

# now predict stuff

predictions = model.predict(x_test)

print('our class predictions on our first test sample are:\n', predictions[0])

print('the most likely class is', np.argmax(predictions[0]))


