# trains a neural network to classify images of clothing

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# import the fashion data set fron MNIST
fashion_mnist = keras.datasets.fashion_mnist

# load the data as numpy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

##images are 28x28 grayscale, labels are 0 through 9, corresponding to clothing type
class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

## preprocess the images
# but first, inspect one just to have a look
plt.figure()
plt.imshow(train_images[0])
plt.show()

# scale values to 0-1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# plot a few again after scaling
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# now set up the keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # flatten 28x28 image into column matrix
    keras.layers.Dense(128, activation=tf.nn.relu), # single layer of 128 nodes
    keras.layers.Dense(10, activation=tf.nn.softmax) # softmax output layer, 10 classes
])

# now select a loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']  # fraction of images correctly classified
              )

## train the model
model.fit(train_images, train_labels, epochs=5)


## evaluate our model performance
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Our test accuracy", test_accuracy)

## get the specific predictions for each test case
predictions = model.predict(test_images)

# predictions is a list of arrays.
# predictions[0] is a 10 element array containing softmax outputs 

print("Our predicted class for test_image[0] is: ", np.argmax(predictions[0]))


## lets plot several test images and their predictions
def plot_image(prediction_array, true_label, image):
  plt.imshow(image, cmap=plt.cm.binary)

  predicted_label = np.argmax(prediction_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(prediction_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(prediction_array, true_label):
  thisplot = plt.bar(range(10), prediction_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(prediction_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

## plot one image to test our plot functions
i = 12
plt.figure(figsize=(6,3)) 
plt.subplot(1,2,1)
plot_image(predictions[i], test_labels[i], test_images[i])

plt.subplot(1,2,2)
plot_value_array(predictions[i], test_labels[i])
plt.show()

# plot the first 5 test images, their predicted label and true label
# correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plt.xticks([]) # don't show x-y scales
  plt.yticks([])
  plot_image(predictions[i], test_labels[i], test_images[i])
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(predictions[i], test_labels[i])
plt.show()


# tf.keras models are optimized to make predictions on a batch, or collection,
# of examples at once. So even though we're using a single image, we need to 
# add it to a list:

# Add the image to a batch where it's the only member.
image = (np.expand_dims(test_images[0],0))

print(image.shape) #(1, 28, 28)

#Now predict the image:
predictions_single = model.predict(image)
print(predictions_single)

plot_value_array(predictions_single[0], test_labels[0])
_ = plt.xticks(range(10), class_names, rotation=45)