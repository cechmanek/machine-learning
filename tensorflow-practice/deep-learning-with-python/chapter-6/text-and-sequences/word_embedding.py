'''
 using word embedding layers in our model of imdb sentiment classification
'''

from tensorflow.keras.datasets import imdb
import tensorflow.keras.preprocessing as preprocess
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Sequential

example_embedding_layer = Embedding(1000, 64) # 1000 tokens max, using 64 dimensional space
# embedding_layer works like a dictionary {int : vector} where the int key is a word index

max_features = 10000 # only consider 10000 most common words
max_length = 20 # we'll truncate or extend our reviews to 20 words


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# data is loaded as integers. x_train[0] = [1,34,256,724,246,21,7,...] or something like this

x_train = preprocess.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = preprocess.sequence.pad_sequences(x_test, maxlen=max_length)


# build a super simple one layer model
model = Sequential()

#model.add(example_embedding_layer) # this is probably bigger than we need
model.add(Embedding(10000, 8, input_length=max_length)) # use a smaller space of 8 dimensions
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

print(history.history)




