'''
 example of tokenizing text of imdb movie reviews as a preprocessing step for word embedding
 to run this example first download the data by visiting http://mng.bz/0tIo
 you also need to download the glove.6b.zip embedding weights from nlp.stanford.edu/projects/glove
'''

import os
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten

# load in all the movie reviews as strings with corresponding labels
imdb_dir = 'aclImdb'
train_dir = os.path.join(imdb_dir ,"train")

labels = []
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir, label_type)

    for file_name in os.listdir(dir_name):
        if file_name[-4:] == '.txt':
            current_file = open(os.path.join(dir_name, file_name))
            texts.append(current_file.read())
            current_file.close()
            
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1) 


# set up the data for tokenization
max_length = 100 # max review length we'll truncate to
num_training_samples = 200
num_validation_samples = 10000
max_words = 10000 # consider only 10000 most common words

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts) # transform after we've fit to it

# get the dict that maps words to numbers
word_index = tokenizer.word_index
print("found %s unique tokens (aka unique words)" %len(word_index))


# get sequences into uniform length via zero padding their ends
sequences = pad_sequences(sequences, maxlen=max_length)
labels = np.asarray(labels)

# since we read in the data as all neg then all pos we need to shuffle it
indices = np.arange(sequences.shape[0]) # make a list of indices
np.random.shuffle(indices)
sequences = sequences[indices]
labels = labels[indices] # make sure sequences and labels are still aligned

# separate data into train, validation, test
x_train = sequences[:num_training_samples]
y_train = labels[:num_training_samples]

x_validation = sequences[num_training_samples: num_training_samples + num_validation_samples]
y_validation = labels[num_training_samples: num_training_samples + num_validation_samples]

x_test = sequences[num_training_samples + num_validation_samples:]
y_test = labels[num_training_samples + num_validation_samples:]


# load GloVe word embedding weights
embeddings_index = {}
embed_file = open('glove.6B.100d.txt')
for line in embed_file:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefficients
embed_file.close()

print("found %s word vectors in GloVe weights" % len(embeddings_index))

embedding_dim = 100 # file we loaded has 100 dimensions
# need to turn embeddings_index into 2D array
# each row is the vector of 100 coefficients corresponding to the word
# ex first_row = [2,4,5,2,5,5,...] for first_word=='hollywood' or something like that

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_words:
        break

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        # only fill in most common words seen. other rows default to all zeros [0,0,0,..]

# now build a model using this GloVe embedding. Use similar architecture as in word_embedding.py 
model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=max_length))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# manually load the GloVe weights into our embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False # no need to retrain this layer

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                     validation_data=(x_validation, y_validation), 
                     epochs=10,
                     batch_size=32)

model.save_weights('pretrained_glove_model.h5')

# plot the history
import matplotlib.pyplot as plt

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(accuracy) + 1)

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title("ACCURACY")
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("LOSS")
plt.legend()

plt.show()

