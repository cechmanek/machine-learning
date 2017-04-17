
# importing the scikit-learn library
# and loading data with it

import sklearn as sk
from sklearn import datasets

from matplotlib import pyplot as plot
import numpy as np

# there are some built in data sets that can be called
iris = datasets.load_iris()
digits = datasets.load_digits()

# grayscale images od 40 differnent people, 10 per person
#olivettiFaces = sk.datasets.fetch_olivetti_faces

# newsgroups text
# texts = sk.datasets.fetch_20newsgroups # loads raw texts
# to get vectorized features you can use:
# sk.feature_extraction.text.CountVectorizer(texts, ,)
# or to get already vectorized features
# textFeat = sk.datasets.fetch_20newsgroups_vectorized

# the .data property returns a n_samples by m_features array
print(digits.data)
print('--------')
print(iris.data)

# the .target property returns 'the ground truth'
# which is used for supervised learning
print(digits.target)
print('______')
print(iris.target)

'''
digits.target gives the ground truth for the digit dataset,
 that is the number corresponding to each digit image 
 that we are trying to learn:
'''

# data is always n_samples by m_features,
# even if original data wasn't
# ex: digits data set is a set of 8x8 images
print(digits.images[0])

myImg = np.uint8(digits.images[0])
plot.imshow(myImg)
plot.show()

# loading external data
# works for numpy arrays,
# scipy sparse arrays,
# pandas dataframes

 '''
pandas.io provides tools to read data from common formats
 including CSV, Excel, JSON and SQL.
DataFrames may also be constructed from lists of tuples
 or dicts.
Pandas handles heterogeneous data smoothly and provides
 tools for manipulation and conversion into a numeric array
 suitable for scikit-learn.
scipy.io specializes in binary formats often used in 
 scientific computing context such as .mat and .arff
numpy/routines.io for standard loading of columnar data
 into numpy arrays
scikit-learn’s datasets.load_svmlight_file for the
 svmlight or libSVM sparse format
scikit-learn’s datasets.load_files for directories of text
 files where the name of each directory is the name of 
 each category and each file inside of each directory 
 corresponds to one sample from that category
'''

'''
For some miscellaneous data such as images, videos, and
 audio, you may wish to refer to:
skimage.io or Imageio for loading images and videos to 
numpy arrays
scipy.misc.imread (requires the Pillow package) to load
 pixel intensities data from various image file formats
scipy.io.wavfile.read for reading WAV files into a numpy
 array
'''


