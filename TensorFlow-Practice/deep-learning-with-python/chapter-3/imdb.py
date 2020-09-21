'''
classifying movie reviews 
An example of binary classification
following the code starting on page 68
slightly modified to use tensorflow's native keras

This specific file fails to train. It throws a segfault. No clue why,
but keras with GPU does work on other examples
'''

import numpy as np

from tensorflow.keras.datasets import imdb


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("I swear to all that is right in this universe I will bring suffering to those responsible for this disgrace to existance!")

# only keep the top 10000 most common words in the dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
'''
print('the shape of our training set is: ', train_data.shape)
print('the first training sample is: ', train_data[0])
print('with corresponding label: ', train_labels[0])
'''
# the data is already in numeric form, we can decode it via built in word index
word_index = imdb.get_word_index()
reverse_word_index = {val: key for key, val in word_index.items()}

decoded_first_sample = [reverse_word_index.get(i-3, '?') for i in train_data[0]]
# first 3 indices are reserved for 'padding', 'start of sequence' and 'unknown'
#print(decoded_first_sample)

# we'll one-hot encode our data samples. this can be done manually. The book doesn't mention a built in method, which is a tragedy

def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# also vectorize labels

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# now it's time to build a model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # binary classification

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

print('\n\n\n')
print('compiled model')
print('\n\n\n')

# training time
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

print('\n\n\n')
print('trained model')
print('\n\n\n')

history_dict = history.history

# let's look at the model training history with matplotlib
import matplotlib.pyplot as plt

training_loss = history_dict['loss']
validation_loss = history_dict['val_loss']

epochs = range(1,len(training_loss)+1)

plt.plot(epochs, training_loss, 'bo', label='Training Loss')
plt.plot(epochs, validation_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()


