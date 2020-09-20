'''
example of how to share the weights of a node and/or layer in Keras
this follows the example code starting on page 247
'''

from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model

lstm = layers.LSTM(32) # create one LSTM layer

left_input = Input(shape=(None,128))
left_output = lstm(left_input) # pass left_input through lstm layer

right_input = Input(shape=(None,128))
right_output = lstm(right_input) # pass right_input through the same lstm layer

merged = layers.concatenate([left_output, right_output], axis=-1)

predictions = layers.Dense(1, activation='sigmoid')(merged)
 
model = Model([left_input, right_input], predictions)

model.summary() # note that only 1 LSTM is shown in the model

# if we had data we would train like this
#model.fit([left_data, right_data], targets)


# this technique is valuable with systems such as stereo camera pairs
# it makes sense to use the same feature extraction conv layers for each camera

# this technique, along the the functional API means whole models can be considered layers

from tensorflow.keras import applications # import the Xception model architecure

xception_base = applications.Xception(weights=None, include_top=False) # just model arch 

left_input = Input(shape=(250,250,3)) # some 250pixel by 250pixel RGB image
right_input = Input(shape=(250,250,3)) # some 250pixel by 250pixel RGB image

left_features = xception_base(left_input) # use same xception_base model for each camera
right_features = xception_base(right_input)

merged_results = layers.concatenate([left_features, right_features], axis=-1)
