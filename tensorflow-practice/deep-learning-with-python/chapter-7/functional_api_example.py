'''
exploring the keras functional api
following the code examples starting on page 236
'''

from tensorflow.keras import Input, layers

# the basic idea of the functional API. define tensors and functions that take and return tensors
input_tensor = Input(shape=(32,)) # define a tensor of given shape, here a batch_sizex32 tensor
print(input_tensor.shape)

dense = layers.Dense(32, activation='relu') # dense is a callable function

output_tensor = dense(input_tensor) # call dense on the input_tensor. returns a tensor


# a side-by-side example of a Sequental model and the equivalent functional version

from tensorflow.keras.models import Sequential, Model

seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# equivalent functial version
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor) # x is output of this layer
x = layers.Dense(32, activation='relu')(x) # input to this dense layer is previous output, 'x'
output_tensor = layers.Dense(10, activation='softmax')(x) # input is again previous output, 'x'

func_model = Model(input_tensor, output_tensor) # output_tensor knows it's build history

func_model.summary() # looks the same as the seq_model defined above

# from here on it's the same workflow
seq_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
func_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#seq_history = seq_model.fit(x_train, y_train, epochs=10, batch_size=128) # if we had data
#func_history = func_model.fit(x_train, y_train, epochs=10, batch_size=128)



