'''
implementing a simple version of Google's Deep Dream
this follows the code example starting on page 281

Deep Dream is remarkably similar to visualizing convnets from chapter 5
wherein we do gradient ascent on input image to maximize output of a layer
instead here we maximize output of several layers, not just one
'''

# need a pretrained convnet. Inception was in the original paper so we'll use that
import scipy as sc
import numpy as np
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as backend

backend.set_learning_phase(0) # disable training on model as we don't need it

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
model.summary()

# choose which layers we want to maximize the output of
# lower layers lead to geometric patterns, higher layers lead to classes like dogs or birds
layer_contributions = {'mixed2':0.2, # some weighted average chosen rather arbitrarily
                        'mixed3':3.0,
                        'mixed4':2.0,
                        'mixed5':1.5}

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Define the loss.
loss = backend.variable(0.0)
for layer_name in layer_contributions:
    # Add the L2 norm of the features of a layer to the loss.
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = backend.prod(backend.cast(backend.shape(activation), 'float32'))
    #loss += coeff * backend.sum(backend.square(activation[:, 2: -2, 2: -2, :])) / scaling
    # += operator is not supported on this TensorFlow version so do it longhand
    loss = loss + coeff * backend.sum(backend.square(activation[:, 2: -2, 2: -2, :])) / scaling
    

# now we need the gradient ascent process to modify input image
# This holds our generated image
dream = model.input

# Compute the gradients of the dream with regard to the loss.
grads = backend.gradients(loss, dream)[0]

# Normalize gradients.
grads /= backend.maximum(backend.mean(backend.abs(grads)), 1e-7)

outputs = [loss, grads]

fetch_loss_and_grads = backend.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

# we'll need some helper functions to handle the image

def resize_image(img, size):
    img = np.copy(img) # local copy to not modify original
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return sc.ndimage.zoom(img, factors, order=1)
    
def save_image(img, filename):
    #pil_img = deprocess_image(np.copy(img)) # not a PIL image, Chollet you fucking moron
    #sc.misc.imsave(filename, pil_img) # imsave() removed years ago. Eat shit Chollet 
    img = deprocess_image(np.copy(img))
    image.save_img(filename, img)
    

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    if backend.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[1]))
        x = x.transpose((1,2,0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x


# heart of Deep Dream algorithm involves processing input image at different scales called octaves
# after each processing step we re-inject the details of original image that were lost
# this keeps the image details sharp

step = 0.01 # gradient ascent step size
num_octaves = 3
octave_scale = 1.4 # upsize image by 40% each time
iterations = 20 # gradient ascent iterations at each scale

max_loss = 10.0 # if loss>10 interupt gradient ascent to stop weird artifacts

base_image_path = 'outdoor_cat.jpg'
cat_image = preprocess_image(base_image_path)

original_shape = cat_image.shape[1:3] # first col is batch size
successive_shapes = [original_shape]
for i in range(1, num_octaves):
    shape = tuple([int(dim / (octave_scale ** 1)) for dim in original_shape])
    successive_shapes.append(shape) # list of shape tuples to run gradient ascent at

successive_shapes = successive_shapes[::-1] # reverse so they're in increasing order

original_image = np.copy(cat_image)

shrunk_original_image = resize_image(cat_image, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_image(cat_image, shape)
    
    # run gradient ascent to modify input image
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
    
    upscaled_shrunk_original_image = resize_image(shrunk_original_image, shape)
    same_size_original = resize_image(original_image, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_image
    img += lost_detail # add back in detail we lost during scale up

    shrunk_original_image = resize_image(original_image, shape)
    save_image(img, filename='final_dream.png')


