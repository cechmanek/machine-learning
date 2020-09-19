'''
multi-headed models with the keras functional api
following code starting on page 238
'''

from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input
import numpy as np

# we'll build a model that takes 2 input streams, a question and a reference text
# each will go through its own LSTM and be concatenated to output an answer to the question
text_vocab_size = 10000 # how many possible unique characters may be present 
question_vocab_size = 10000
answer_vocab_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text') # first input head. reference text

embedded_text = layers.Embedding(64, text_vocab_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question') # second input head. question

embedded_question = layers.Embedding(64, question_vocab_size)(question_input)
encoded_question = layers.LSTM(32)(embedded_question)

# now combine these two LSTM nodes via concatenation
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

answer = layers.Dense(answer_vocab_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer) # two inputs and one output
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# prepare for training
num_samples = 1000
max_length = 100 # length of each sample. here it's 100 characters

text = np.random.randint(1, text_vocab_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocab_size, size=(num_samples, max_length))
answer = np.random.randint(0,1, size=(num_samples, answer_vocab_size)) # each answer is only one word

# we have two options for passing data to multi-headed networks:
model.fit([text, question], answer, epochs=10, batch_size=64) # pass a list of inputs to x_train
# OR
training_dict  = {'text':text, # names are optionally defined when calling Input() above
                  'question':question}
#model.fit(training_dict, answer, epochs=10, batch_size=2) # pass a dict with key=name of input

'''
It looks like passing and dict uses considerably more GPU memory.
using the first method, [text, question] i can train on batch sizes of 64.
using dict method max batch size == 2
'''
