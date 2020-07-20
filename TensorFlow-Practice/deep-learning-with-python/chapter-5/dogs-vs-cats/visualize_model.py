'''
visualizing deep net of our dog-cat classifier
following the example on page 160
'''

from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
import os

model = models.load_model('cats_and_dogs_small_2.h5')
model.summary() 

# grab an image not in our training set
image_path = os.path.join('dogs-vs-cats-small','test','cats','cat.1700.jpg')

cat_image = image.load_img(image_path, target_size=(150,150))
image_tensor = image.img_to_array(cat_image)
image_tensor = np.expand_dims(image_tensor, axis=0) # need to add an extra axis
image_tensor /= 255 # normalize, as our model was trained on nomralized input

plt.imshow(image_tensor[0]) # ignore added first dim when viewing
#plt.show()

# METHOD 1: look at output activations that occur for a specific input image
# collect the output values of the first 8 layers of our trained model
layer_outputs = [layer.output for layer in model.layers[:8]]

# create a model that will return these layer_outputs as it's output
# this implicitly grabs the whole trained model we loaded earlier
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(image_tensor) # returns list of 8 np arrays. 1 per layer

print('we have {} layers of activations'.format(len(activations)))

# we can view these filters as 2d images
plt.matshow(activations[0][0,:,:,0], cmap='viridis')# first layer, first activation
plt.matshow(activations[0][0,:,:,1], cmap='viridis')# first layer, second activation
plt.matshow(activations[0][0,:,:,2], cmap='viridis')# first layer, third activation

plt.matshow(activations[2][0,:,:,2], cmap='viridis')# third layer, third activation
#plt.show()

# lets view them all together
layer_names = []
for layer in model.layers[:8]:
  layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
  n_features = layer_activation.shape[-1]
  size = layer_activation.shape[1]
  
  num_cols = n_features // images_per_row # number of cols in mosaic image
  display_grid = np.zeros((size * num_cols, images_per_row * size))
  
  for col in range(num_cols):
    for row in range(images_per_row):
      channel_image = layer_activation[0,:,:,col * images_per_row + row]
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std() 
      channel_image *= 64
      channel_image += 128 
      channel_image = np.clip(channel_image,0, 255).astype('uint8') # threshold values 
      
      display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image 


  scale = 1.0 / size
  plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid, aspect='auto', cmap='viridis')
#plt.show()


# METHOD 2:  we can visualize what a given filter is responding to
# we do this by gradient ascent on input image until filter output is maximized

# use vgg16 as its filters are much more finely tuned than our custom net
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as backend
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1' # pick an arbitrary layer to visualize
filter_index = 0 # pick an arbitrary filter in that layer to visualize

layer_output = model.get_layer(layer_name).output

loss = backend.mean(layer_output[:,:,:, filter_index])

grads = backend.gradients(loss, model.input)[0] # .gradients always returns list of tensors

# normalize with L2 norm for smoother gradient decent. Not strictly needed, but is beneficial
grads /= (backend.sqrt(backend.mean(backend.square(grads))) + 1e-5) # +1e-5 to avoid div0 error

# we need to iteratively update the input image now. tensorflow has a tool for that
iterate  = backend.function([model.input], [loss, grads])

# 'iterate' object takesi an image and [loss function, grad function], outputs loss & grad value
# example: loss_value, grads_value = iterate([np.zeros((1,150,150,3))])  

# loop until the input image converges to maximize the filter output
starting_image = np.random.random((1,150,150,3)) *20 + 128 # start with mean 128 uniform dist 20
step_size = 1.0
for i in range(40):
  loss_value, grad_value = iterate([starting_image])

  starting_image += grad_value * step_size # gradient ASCENT to MAXIMIZE loss

# starting_image may not be in the range (0,255) so process it to clean it up
img = starting_image - starting_image.mean()
img /= (img.std() + 1e-5) # to avoid div0 errors
img *= 0.1 # scales down image to have max(abs(img)) <= 0.1
img += 0.5
img = np.clip(img, 0, 1)
img *= 255

plt.figure()
plt.imshow(img[0].astype('uint8'))
plt.title("input that maximizes {} filter activation".format(layer_name))
#plt.show()


# METHOD 3: viewing areas of input images that most strongly contribute to class prediction 

# reload VGG16 but this time with the dense layers
model = VGG16(weights='imagenet')

# vgg16 has some custom preprocessing steps, luckily keras has them built in so import them
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# use an image from imagenet classes
img = image.load_img('two_elephants.jpeg', target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

img = preprocess_input(img)

predictions = model.predict(img)
print("we predicted", decode_predictions(predictions, top=3)[0])
# we should have predicted 'african elephant' which should be ouput #386


elephant_output = model.output[:,386]

last_conv_layer = model.get_layer('block5_conv3')

grads = backend.gradients(elephant_output, last_conv_layer.output)[0]
# gradient of elephant class wrt the output feature map of block5_conv3

pooled_grads = backend.mean(grads, axis=(0,1,2))
# vector of shape (512,). each val is mean intensity of grad over specific feature map channel

iterate = backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([img])

# multiply each channel in feature map arr by 'how important this channel is' wrt elephant class
for i in range(512):
  conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

# normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# we can view the heatmap on it's own

plt.matshow(heatmap)
plt.title("heatmap activation for aficant elephant")
#plt.show()

from matplotlib import image as pltimage
pltimage.imsave('heatmap.jpeg', heatmap)

''' this section fails as we are lacking memory
# use opencv to overlay heatmap with the original image
import cv2

elephants = cv2.imread('two_elephants.jpeg')
heatmap = cv2.resize(heatmap, (elephants.shape[1], elephants.shape[0]))
heatmap = np.uint8(255*heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed = heatmap * 0.4 + image

cv2.imshow(superimposed,'heatmap on image')
cv2.waitKey()
'''
