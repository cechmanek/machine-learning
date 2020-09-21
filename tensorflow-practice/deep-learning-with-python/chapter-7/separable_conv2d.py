'''
showing the use of separable 2d convolutional layers as a replacement for conv2d layers
this follows the code examples starting on page 262

For more info on why depthwise separable convolutions are superior, read:
"Xception: Deep Learning with Depthwise Separable Convolutions"
which introduced them
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

# build a simple 10 class image classifier
height = 64
width = 64
channels = 3
num_classes = 10

model = Sequential()
model.add(layers.SeparableConv2D(32, 3, activation='relu', input_shape=(height, width, channels)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
