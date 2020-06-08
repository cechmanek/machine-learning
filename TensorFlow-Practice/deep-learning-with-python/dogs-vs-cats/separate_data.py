'''
large scale deep convnet classifier
following the example starting on page 132
'''

import os, shutil

# manage data locations
original_data_dir = os.getcwd() + '/train'

base_dir = os.getcwd() + '/dogs-vs-cats-small'
try:
  os.mkdir(base_dir)
except FileExistsError:
  pass

train_dir = os.path.join(base_dir, 'train')
try:
  os.mkdir(train_dir)
except FileExistsError:
  pass

validation_dir = os.path.join(base_dir, 'validation')
try:
  os.mkdir(validation_dir)
except FileExistsError:
  pass

test_dir = os.path.join(base_dir, 'test')
try:
  os.mkdir(test_dir)
except FileExistsError:
  pass


train_cats_dir = os.path.join(train_dir, 'cats')
try:
  os.mkdir(train_cats_dir)
except FileExistsError:
  pass


train_dogs_dir = os.path.join(train_dir, 'dogs')
try:
  os.mkdir(train_dogs_dir)
except FileExistsError:
  pass


validation_cats_dir = os.path.join(validation_dir, 'cats')
try:
  os.mkdir(validation_cats_dir)
except FileExistsError:
  pass


validation_dogs_dir = os.path.join(validation_dir, 'dogs')
try:
  os.mkdir(validation_dogs_dir)
except FileExistsError:
  pass


test_cats_dir = os.path.join(test_dir, 'cats')
try:
  os.mkdir(test_cats_dir)
except FileExistsError:
  pass


test_dogs_dir = os.path.join(test_dir, 'dogs')
try:
  os.mkdir(test_dogs_dir)
except FileExistsError:
  pass


# copy the first 1000 cat images into train_cats_dir
fnames = {'cat.{}.jpg'.format(i) for i in range(1000)}
for fname in fnames:
  src = os.path.join(original_data_dir, fname)
  dst = os.path.join(train_cats_dir, fname)
  shutil.copyfile(src,dst)

# copy the next 500 cat images into validation_cats_dir
fnames = {'cat.{}.jpg'.format(i) for i in range(1000, 1500)}
for fname in fnames:
  src = os.path.join(original_data_dir, fname)
  dst = os.path.join(validation_cats_dir, fname)
  shutil.copyfile(src,dst)

# copy the next 500 cat images into test_cats_dir
fnames = {'cat.{}.jpg'.format(i) for i in range(1500, 2000)}
for fname in fnames:
  src = os.path.join(original_data_dir, fname)
  dst = os.path.join(test_cats_dir, fname)
  shutil.copyfile(src,dst)

# repeat the copying for dog images

# copy the first 1000 cat images into train_dogs_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1000)}
for fname in fnames:
  src = os.path.join(original_data_dir, fname)
  dst = os.path.join(train_dogs_dir, fname)
  shutil.copyfile(src,dst)

# copy the next 500 cat images into validation_dogs_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1000, 1500)}
for fname in fnames:
  src = os.path.join(original_data_dir, fname)
  dst = os.path.join(validation_dogs_dir, fname)
  shutil.copyfile(src,dst)

# copy the next 500 cat images into test_dogs_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1500, 2000)}
for fname in fnames:
  src = os.path.join(original_data_dir, fname)
  dst = os.path.join(test_dogs_dir, fname)
  shutil.copyfile(src,dst)


# sanity check that we have the right number of files in each category
print('training cat images: ', len(os.listdir(train_cats_dir)))
print('training dog images: ', len(os.listdir(train_dogs_dir)))
print('validation cat images: ', len(os.listdir(validation_cats_dir)))
print('validation dog images: ', len(os.listdir(validation_dogs_dir)))
print('testing cat images: ', len(os.listdir(test_cats_dir)))
print('testing dog images: ', len(os.listdir(test_dogs_dir)))
