'''
Deep convnet for dog vs cat classifier
following the example on page 134
'''
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# manage data locations
original_data_dir = os.path.join(os.getcwd(), 'train')

base_dir = os.path.join(os.getcwd(), 'dogs-vs-cats-small')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# define and compile model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu')) 
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

_ = input('press enter key to continue')

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
  batch_size=10, # jetson gpu memory is limited to batches of 10
  class_mode='binary') # assigns binary labels to images

validation_generator = test_datagen.flow_from_directory(
  directory=validation_dir,
  target_size=(150,150),
  batch_size=10,
  class_mode='binary')

# view some of the training and validation iamges
try:
  num_images = int(input('enter the number of images you want to view '))
except ValueError:
  print('skipping viewing')
  num_images = 0

i = 0
batch_num = 0
label_dict = {0:'cat', 1:'dog'}
for batch in train_generator:
  if i >= num_images:
    break

 images = batch[0]
  labels = batch[1]
  for image, label in zip(images, labels):
    plt.imshow(image)
    plt.title('image {} from batch {}, labelled {}'.format(i, batch_num, label_dict[label]))
    plt.show()
    i += 1
    if i >= num_images:
      break
  batch_num += 1

# generators yield batches of 10 indefinitely, so 200 steps/epoch gives 2000 images per epoch 
history = model.fit(train_generator, 
                    steps_per_epoch=200,
                    epochs=100, 
                    validation_data=validation_generator, 
                    validation_steps=30)

model.save('cats_and_dogs_small_2.h5')

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
