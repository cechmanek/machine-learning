'''
generative adversarial networks used to create images from the CIFAR dataset, specifically frogs
this code follows the example starting on page 308
'''

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

latent_dims = 32
height = 32
width = 32
channels = 3 # RGB images

# start from a random vector of size 32 
generator_input = layers.Input(shape=(latent_dims))

# transform into a 32*32 pixel, RGB image to match CIFAR image size
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(25, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x) # upsamples to 32x32 pixels
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()

# now build the discriminator

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128,3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x) # binary classification if image is real or generated

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary() # essentially a binary image classifier

# GANs are notoriously difficult to train, so use some good heuristics for optimizer
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# while training the full GAN we freeze the discriminator weights
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dims))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
gan.summary()

# now we implement the full GAN training cycle. For a good explaination read pg 310

import os
from tensorflow.keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train[y_train.flatten() == 6] # select only frog images, label==6

x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32')/255.0

iterations = 10000
batch_size = 20
save_dir = 'gan_frog_images'

start = 0

for step in range(iterations):
    # generate fake images 
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dims))
    generated_images = generator.predict(random_latent_vectors)

    # mix real and fake images into one batch
    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size,1)), np.zeros((batch_size,1))])
    labels + 0.05 * np.random.random(labels.shape) # add random noise to labels, a helpful trick
    # zero-labelled are real, one-labelled are generated images


    # train discriminator to distinguish between real and fake images
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # train generator through our gan, where we say all images are real
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dims))
    misleading_targets = np.zeros((batch_size, 1))
   
    # discriminator weights are frozen when run through our gan model 
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size # update our pointers so we grab a new set of real images next step
    if start > len(x_train) - batch_size:
        start = 0 # go back to beginning of real image dataset

    if step % 100 == 0:
        gan.save_weights('gan.h5')
        # print an update, and save some images for comparison afterward
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)

        fake_img = image.array_to_img(generated_images[0] * 255.0, scale=False)
        fake_img.save(os.path.join(save_dir, 'generated_frog_' + str(step) + '.png'))

        real_img = image.array_to_img(real_images[0] * 255.0, scale=False)
        real_img.save(os.path.join(save_dir, 'real_frog_' + str(step) + '.png'))

