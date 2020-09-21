'''
deep convnet classifier using VGG16 backend
following the code example beginning on page 145
and modified to use built in tensorflow.keras
and run on a jetson tx2
'''

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# manage data locations
original_data_dir = os.path.join(os.getcwd(), 'train')
base_dir = os.path.join(os.getcwd(), 'dogs-vs-cats-small')

train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_dir = os.path.join(base_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_dir = os.path.join(base_dir, 'test')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# import the pretrained VGG convolutional layers
conv_base = VGG16(weights='imagenet', # which dataset VGG16 is trained on
		  include_top=False, # wether or not to include dense layers
		  input_shape=(150,150,3)) # optional argument

conv_base.summary()
_ = input('press enter to continue')

conv_base.trainable = False # freeze these weights 

# now build just the dense section of the model to be added ontop the conv_base

model = Sequential()
model.add(conv_base) # yes, it really is this simple
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # for binary classification

model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy']) 

model.summary()
_ = input('press enter to continue')

# use data augmentation for training, as we did in cat-vs-dog.py

# preprocess and augment the images for training
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=40, # range in degrees 0->180
                                    width_shift_range=0.2, # as a fraction of image size
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255) # no need to synthesize test data, but we could

train_generator = train_datagen.flow_from_directory(
  directory=train_dir, # automatically figures out which image class based on sub folders
  target_size=(150,150),
  batch_size=5, # jetson gpu memory is limited to batches of 5 for this deep convnet 
  class_mode='binary') # assigns binary labels to images

validation_generator = test_datagen.flow_from_directory(
  directory=validation_dir,
  target_size=(150,150),
  batch_size=5,
  class_mode='binary')

# training time, should be decently quick as convolution layers are frozen

history = model.fit(train_generator, 
	epochs=100, 
	steps_per_epoch=200, 
	validation_data=validation_generator, 
	validation_steps=30)

# plot training history
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)
plt.figure()
plt.title('training and validation accuracy')
plt.plot(epochs, accuracy, 'b-', label='training accuracy')
plt.plot(epochs, val_accuracy, 'ro', label='validation accuracy')
plt.legend()

plt.figure()
plt.title('training and validation loss')
plt.plot(epochs, loss, 'b-', label='training loss')
plt.plot(epochs, val_loss, 'ro', label='validation loss')
plt.legend()

plt.show()
