# datasets

from sklearn import datasets

iris = datasets.load_iris()

data = iris.data
# iris.data is 150 samples, each with 4 features
print(data.shape)
print(data.dtype)

# reshaping data
digits = datasets.load_digits()
data = digits.data
# digits.data is 1797 samples of 64 pixels (float64)
imgs = digits.images
# digits.imgs is 1797 samples of 8x8 images (grayscale)
print(data.shape)
print(data.dtype)
from matplotlib import pyplot as plot
plot.imshow(imgs[0], cmap='gray')
#plot.show()

# to convert images to processable data reshape into vectors
from cv2 import imread
img = imread('Messi.png',0)

imgVector = img.reshape((img.shape[0]*img.shape[1], -1))

print(img.shape)
print(imgVector.shape)

print(img[0,89])
print(imgVector[89])

