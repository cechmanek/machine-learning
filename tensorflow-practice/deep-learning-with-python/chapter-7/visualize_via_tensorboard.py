'''
using tensorboard html visualizer to monitor training on a 1D convnet for imdb sentiment
following the code examples starting on page 253
'''

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 2000
max_length = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, max_length)
x_test = sequence.pad_sequences(x_test, max_length)

model = keras.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_length, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1, activation='sigmoid')) # 1 output for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# now create a tensorboard callback and have it write logs to the 'logs' directory we created

callbacks = [keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1, embeddings_data=x_train)]

history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)

# launch the TensorBoard html viewer through the command line:
# tensorboard --logdir=logs
# and visit localhost:6006 in your browser
