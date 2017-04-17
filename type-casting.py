# conventions for type casting

# unless told otherwise input is cast to float64


import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)

x = rng.rand(10,2000)
x = np.array(x, dtype = 'float32')

print(x.dtype)

# sklearn method auto casts to float64
transformer = random_projection.GaussianRandomProjection()
newX = transformer.fit_transform(x)
print(newX.dtype) 

# regression targets are cast, classification targets aren't
from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()

clf.fit(iris.data, iris.target)

prediction = list(clf.predict(iris.data[:3])) 
print(prediction)
print(prediction[0].dtype)


# bulding a classification model instead
# target are not cast to float64
clf.fit(iris.data,iris.target_names[iris.target])

classification = clf.predict(iris.data[:3])
print(classification)
print(classification[0].dtype)







