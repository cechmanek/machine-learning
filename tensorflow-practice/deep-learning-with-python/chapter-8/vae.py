'''
variational autoencoder for image generation
this follows the code examples starting on page 300
'''

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as backend
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np

img_shape = (28,28,1) # start with a small grayscale image
batch_size = 16
latent_dims = 2 # to dimensions for grayscale image, a 2D plane

input_img = keras.Input(shape=img_shape)

# now build a simple convnet via functional API

x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2,2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

shape_before_flattening = backend.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dims)(x)
z_log_variance = layers.Dense(latent_dims)(x) # multi-headed

# input image is being encoded into a mean and log_variance.
# together these 2 vectors define the latent space of the images in this encoder

# we need a custom sampling layer to draw from this distribution defined by mean and log_variance
# custom layers in Keras are defined via the layers.Lambda class

def sampling(args):
    z_mean, z_log_var = args
    epsilon = backend.random_normal(shape=(backend.shape(z_mean)[0], latent_dims), mean=0., stddev=1.)
    return z_mean + backend.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_variance])

# above we defined the encoder part, now define the decoder model

decoder_input = layers.Input(backend.int_shape(z)[1:])

# upsample the input since the encoder downsampled it
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x) # undoes flatten() layer in encoder

# conv2dtranspose to decode z into a feature map the same size as input image
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)

z_decoded = decoder(z)


# we need a custom layer to handle the dual losses of a VAE 
from CustomVariationalLayer import CustomVariationalLayer
y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_variance])

# we can now compile and train our model on mnist

vae = Model(input_img, y)
vae.compile(optimizer='adam', loss=None) # we have a custom loss, so pass None here
vae.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_train = x_train.reshape(x_train.shape + (1,))

x_test = x_test.astype('float32') / 255
x_test = x_test.reshape(x_test.shape + (1,))

batch_size = 64

vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None)) 

# plot our results

import matplotlib.pyplot as plt
from scipy.stats import norm

n = 15 # display a grid of 15 by 15 digit images
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        
        digit = x_decoded[0].reshape(digit_size, digit_size) # because x_decoded is a vector
        figure[i * digit_size: (i+1) * digit_size, j * digit_size: (j+1) * digit_size] = digit

plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.title('digits uniformly sampled from VAE latent space')
plt.show()

