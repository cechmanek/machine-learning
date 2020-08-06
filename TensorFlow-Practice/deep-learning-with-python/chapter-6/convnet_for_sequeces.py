'''
using 1D convolutional nets on sequential data
following the code example starting on page 226
'''

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, GlobalMaxPool1D, Dense
import matplotlib.pyplot as plt

max_features = 10000
max_length = 500

print('loading data')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_length))
model.add(Conv1D(32, 7, activation='relu')) # 32 filters of sizie 7 each
model.add(MaxPool1D(5)) # pool over a window of size 5
model.add(Conv1D(32, 7, activation='relu')) # 32 filters of sizie 7 each
model.add(GlobalMaxPool1D())
model.add(Dense(1))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

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

