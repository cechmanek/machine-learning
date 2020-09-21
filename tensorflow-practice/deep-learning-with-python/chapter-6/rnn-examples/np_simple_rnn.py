'''
 simple feedback recurrent nueral network implemented in numpy
'''

''' here's the basic structure of having a state feeback look into a function block 'f'
state_t = 0 # state at time 't'

for input_t in input_sequence:
    output_t = f(input_t, state_t) # output at time 't'
    state_t = output_t
'''

import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))

state_t = np.zeros((output_features,))

W = np.random.random((output_features, input_features)) # weight matrix on input vector
U = np.random.random((output_features, output_features)) # weight matrix on state vector
b = np.random.random((output_features,)) # bias vector

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) # activation function==tanh
    successive_outputs.append(output_t)

    state_t = output_t

final_output_sequence = np.concatenate(successive_outputs, axis=0)
