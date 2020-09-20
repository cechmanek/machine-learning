'''
using LSTMs to generate sequences of text
this code follows the example starting on page 273
'''

import numpy as np
import random
import sys
import tensorflow.keras as keras
from tensorflow.keras import layers

# define some helper functions for sampling the output of LSTM

# temperature quantifies entropy of output. low temperature=low randomness
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(originial_disribution) / temperature # original dist is 1D np array 
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution) # sum of distribution may not be 1 so renormalize

# sample one characeter from softmax output. this includes reweighting distribution
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(preds) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / (np.sum(exp_predictions) + 1e-6) # add epsilon to keep sum(p)<1
    
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

path = keras.utils.get_file('nietzsche.txt',
                             origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path).read().lower()
print('nietzsche corpus length:', len(text))

# extract overlapping sequences of length=max_length to be used as training samples for our LSTM
max_length = 60
step = 3 # sample new sequence every 3 chars. ex: 'abcdefghij' ->['abcd', 'defg','ghij']

sentences = []
next_chars = [] # the chars following each sentence. the targets to train on

for i in range(0, len(text) - max_length, step):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])

print('Number of sequences we are training on:', len(sentences))

chars = sorted(list(set(text))) # our alphabet. may include punctuation, numbers and symbols
print('Unique characters:', len(chars))

char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorizing text sentences into numerial vectors...')
x = np.zeros((len(sentences), max_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) 

# one hot encode characters into binary arrays
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    
    y[i, char_indices[next_chars[i]]] = 1    


# now build LSTM model

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(max_length, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')

# repeatedly train and sample model as we train
for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x,y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text) - max_length -1)
    generated_text = text[start_index: start_index + max_length] # some random valid text to start
    print('--- Generating with seed: ', generated_text)

    # sample model after each epoch of training, and at a range of temperatures
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # generate 400 characters as output for this temp and at this epoch
        for i in range(400):
            sampled = np.zeros((1, max_length, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
