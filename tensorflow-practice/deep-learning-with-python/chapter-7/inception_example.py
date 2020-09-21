'''
How to build an inception-style network with the keras functional api
this code follows the examples starting on page 243

Be sure to look at the graphical image on page 243
'''

from tensorflow.keras import Model
from tensorflow.keras import layers

x = layers.Input(shape=(512,512,64,))# some 4D tensor, typically created by a convolutional filters 

# all branches receive the same input tensor x
branch_a = layers.Conv2D(128, 1, activation='relu', padding='same', strides=2)(x) # strides across spacial dimension

branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3, padding='same', strides=2)(x) # strides occur across pooling dimension
branch_c = layers.Conv2D(128, 3, padding='same', activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, padding='same', activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(branch_d)

# all branches a,b,c,d, are paralell so combine them at the end
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
