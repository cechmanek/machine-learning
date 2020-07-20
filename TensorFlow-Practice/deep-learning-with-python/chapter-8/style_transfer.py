'''
implementing neural style transfer
following the code examples starting on page 287
'''

import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend 
import tensorflow.keras as keras

# set up some helper functions for image pre and post processing
def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def deprocess_image(x):
    x[:,:,0] += 103.939 # zero center by removing mean pixel value from ImageNet
    x[:,:,1] += 116.779 # vgg19.preprocess substracts these so we need to add them back
    x[:,:,2] += 123.68
    x = x[:,:,::-1] # convert from BGR to RGB, since vgg19.preprocess did the reverse conversion
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# we also need functions for style and content losses
def content_loss(base, combination):
    return backend.sum(backend.square(combination - base)) # L2 norm of 2 images, pixel by pixel

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2,0,1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram
    
def style_loss(style, combination, img_height, img_width):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    # L2 norm of style-content gram matrices, normalized by image size, aka num scalars in image
    return backend.sum(backend.square(S-C)) / (4.0 * (channels**2) * (size**2))

def total_variation_loss(x, img_height, img_width):
    # total variation is a measure of 'smoothness' in image. It's regularization to stop pixelation
    a = backend.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = backend.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))
    
target_image_path = 'castle.jpg'
style_reference_image_path = 'lava.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# create 3 images, the 2 references are constant, the combination output is being modified 
target_image = backend.constant(preprocess_image(target_image_path, img_height, img_width))
style_reference_image = backend.constant(preprocess_image(style_reference_image_path, img_height, img_width))
combination_image = backend.placeholder((1, img_height, img_width, 3))

# combine three images into a single batch
input_tensor = backend.concatenate([target_image, style_reference_image, combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

# assign which layers will be used for style, and which single layer will be used for content
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

# assign weights to the importance of style, content, and total variation. Play around with these
style_weight = 1.0
content_weight = 0.025
total_variation_weight = 1e-4

# define the total loss as the weighted sum of style, content and variation lsos
loss = backend.variable(0.0)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]

# add weighted content_loss from single layer
#loss += content_weight*content_loss(target_image_features, combination_features)
loss = loss +  content_weight*content_loss(target_image_features, combination_features)

# add weighted style_loss from multiple layers
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    s_loss = style_loss(style_reference_features, combination_features, img_height, img_width)
    #loss += (style_weight / len(style_layers)) * s_loss
    loss = loss + (style_weight / len(style_layers)) * s_loss

# add total_variation loss
#loss += total_variation_weight * total_variation_loss(combination_image, img_height, img_width)
loss = loss + total_variation_weight * total_variation_loss(combination_image, img_height, img_width)

grads = backend.gradients(loss, combination_image)[0]

# function used inside of Evaluator class, but defined outside it 
# Holy Fucking Hell Chollet is a garbage excuse for a programmer
fetch_loss_and_grads = backend.function([combination_image], [loss, grads])


# for the gradient descent process we will our Evaluator class
from Evaluator import Evaluator
evaluator = Evaluator(img_height, img_width, fetch_loss_and_grads)

# The specific gradient descent algorithm used by Gatys in his paper is L-BFGS
# which scipy has an implementation ofi
from scipy.optimize import fmin_l_bfgs_b
import time

result_prefix = 'my_result'
iterations = 20

x = preprocess_image(target_image_path, img_height, img_width)
x = x.flatten()

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    
    file_name = result_prefix + '_at_iteration_%d.png' % i
    keras.preprocessing.image.save_img(file_name, img)
    print('Image saved as:', file_name)

    end_time = time.time()

    print('Iteration %d completed in %ds' % (i, end_time - start_time))
