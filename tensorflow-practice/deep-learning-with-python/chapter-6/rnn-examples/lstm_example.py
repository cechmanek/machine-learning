''' 
example of using the Keras LSTM RNN layer on imdb data set
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

import matplotlib.pyplot as plt

#prep imdb data
max_features = 10000
max_length = 500
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)


# build a classification model for imdb sentiment analysis
model = Sequential()

model.add(Embedding(10000, 32)) # input shape of 32
model.add(LSTM(32, return_sequences=False)) # there are MANY settings. leave 'em as default 
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

accuracy = history.history['acc']
loss = history.history['loss']
val_accuracy = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy)+1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('LOSS')
plt.legend()

plt.figure()

plt.plot(epochs, accuracy, 'ro', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('ACCURACY')
plt.legend()

plt.show()
