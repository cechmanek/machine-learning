'''
implementing a simple version of Google's Deep Dream
this follows the code example starting on page 281

Deep Dream is remarkably similar to visualizing convnets from chapter 5
wherein we do gradient ascent on input image to maximize output of a layer
instead here we maximize output of several layers, not just one
'''

# need a pretrained convnet. Inception was in the original paper so we'll use that
import scipy as sc
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as backend

backend.set_learning_phase(0) # disable training on model as we don't need it

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# choose which layers we want to maximize the output of
# lower layers lead to geometric patterns, higher layers lead to classes like dogs or birds
layer_contributions = {'mixed2':0.2, # some weighted average chosen rather arbitrarily
                        'mixed3':3.0,
                        'mixed4':2.0,
                        'mixed5':1.5}

layer_dict = {[(layer.name, layer) for layer in lodel.layers]}

loss = backend.variable() # the loss we'll use to minimize

for layer_name in layer_contributions: # .keys()?
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    
    # take L2 norm of the combined layer outputs as the loss 
    scaling = backend.prod(backend.cast(backend.shape(activation), 'float32'))
    loss += coeff * backend.sum(backend.square(activation[:, 2: -2, 2:, -2, :])) / scaling    

# now we need the gradient ascent process to modify input image

dream = model.input

grads = backend.gradients(loss, dream)[0]
grads /= backend.maximum(backend.mean(backend.abs(grads)), e-7) # normalize gradients

outputs = [loss, grads]

fetch_loss_and_grads = backend.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_value = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_vales -= eval_loss_and_grads(x)
            if max_loss is not None and loss_value > max_Loss:
                break
            print('...Loss value at', i, ':', loss_value)
            x += step * grad_values
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
sucessive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** 1)) for dim in original_shape])
    successive_shapes.append(shape) # list of shape tuples to run gradient ascent at

successive_shapes = successive_shapes[::-1] # reverse so they're in increasing order

original_image = np.copy(cat_image)

shrunk_original_image = resize_image(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_image(cat_image, shape)
    
    # run gradient ascent to modify input image
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
    
    upscaled_shrunk_orginial_image = resize_image(shrunk_original_image, shape)
    same_size_orignial = resize_image(original_image, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_image
    img += lost_detail # add back in detail we lost during scale up

    shrunk_original_image = resize_image(orignial_image, shape)
    save_image(img, filename='final_dream.png')


# we'll need some helper functions to handle the image

def resize_image(img, size):
    img = np.copy(img) # local copy to not modify original
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return sc.ndimage.zoom(img, factors, order=1)
    
def save_image(img, filename):
    pil_img = deprocess_image(np.copy(img))
    sc.misc.imsave(filename, pil_img)

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.image_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    if backend.image_dat_format() == 'channels_first':
        x = x.reshpae((3, x.shape[2], x.shape[1]))
        x = x.transpose((1,2,0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x


