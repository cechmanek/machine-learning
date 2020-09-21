'''
implementing a residual code block in Keras functional API
following the code example starting on page 245
'''

from tensorflow.keras import layers


x = layers.Input(shape=(512,512,128,)) # some 4D input tensor, usually created by a previous layer

# make 3 Conv2D layers and stack them on top of each other. their input is x
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

# add x to the output of the 3 Conv2D layers
output = layers.add([y,x])

'''
graphically, this looks like

x -> Conv -> Conv -> Conv -> (+) -> output
|                             ^
|                             |
 ->-->-->-->-->-->-->-->-->-->

x and last_y must have the exact same shape to add, so padding='same' is needed
'''


# to add x back into the output that has had its shape changed, up/down sample either x or last_y

x = layers.Input(shape=(512,512,128,)) # some 4D input tensor, usually created by a previous layer

z = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
z = layers.Conv2D(128, 3, activation='relu', padding='same')(z)
z = layers.MaxPool2D(2, strides=2)(z) # z is now smaller than x due to pooling

residual = layers.Conv2D(128, 1, strides=2, padding='same')(x) # downsample x before adding

output = layers.add([z, residual])

'''
graphically, this looks like

x -> Conv -> Conv -> MaxPool -> (+) -> output
|                                ^
|                                |
 ->-->--> downSample -->-->-->-->

x and last_z must have the exact same shape to add, so downSample x to match MaxPool z 
'''


