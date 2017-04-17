# example prediction with digits data set

#import sklearn as sk
from sklearn import datasets
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plot
digits = datasets.load_digits()


# create python estimator object using Support Vector Classification
clf = svm.SVC(gamma = 0.001, C = 100)

# train the clf object with our data
clf.fit(digits.data[:-1], digits.target[:-1])
# params(data, correct classification for data)

# used all the last data value, so now test with that value
myPrediction = clf.predict(digits.data[-1:])
print('my prediction is ',myPrediction)

print(digits.data[-1:])

plot.imshow(digits.data[-1:].reshape(8,8))
plot.show()