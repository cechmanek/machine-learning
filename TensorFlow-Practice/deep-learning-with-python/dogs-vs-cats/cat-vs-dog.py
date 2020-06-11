'''
Deep convnet for dog vs cat classifier
following the example on page 134
'''
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# manage data locations
original_data_dir = os.getcwd() + '/train'

base_dir = os.getcwd() + '/dogs-vs-cats-small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Conv2D(32, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Droput(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# preprocess and augment the images for training
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=40, # range in degrees 0->180
                                    width_shfit_range=0.2, # as a fraction of image size
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255) # no need to synthesize test data, but we could

train_generator = train_datagen.flow_from_directory(
  directory=train_dir, # automatically figures out which image is which class based on sub folders
  target_size=(150,150),
  batch_size=10, # jetson gpu memory is limited to batches of 10
  class_mode='binary') # assigns binary labels to images

validation_generator = test_datagen.flow_from_directory(
  directory=validation_dir,
  target_size=(150,150),
  batch_size=10,
  class_mode='binary')

# generators yield batch sizes of 20 indefinitely, so 100 steps per epoch gives 2000 images per epoch 
history = model.fit(train_generator, 
                    steps_per_epoch=100,
                    epochs=30, 
                    validation_data=validation_generator, 
                    validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

# plot training history
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)
plt.figure()
plt.title('training and validation accuracy')
plt.plot(epochs, accuracy, 'bo', label='training accuracy')
plt.plot(epochs, val_accuracy, 'ro', label='validation accuracy')
plt.legend()

plt.figure()
plt.title('training and validation loss')
plt.plot(epochs, loss, 'b-', label='training loss')
plt.plot(epochs, val_loss, 'r-', label='validation loss')
plt.legend()

plt.show()