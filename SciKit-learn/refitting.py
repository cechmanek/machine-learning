# refitting and updating parameters

import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plot

rng = np.random.RandomState(0)
x = rng.rand(100,10)
y = rng.binomial(1,0.5,100)
xTest = rng.rand(4,10)

clf = SVC()
clf.set_params(kernel = 'linear').fit(x,y)
# rbf is default kernel value

print('prediction with linear kernel:')
print(clf.predict(xTest))

# results with different parameters
clf.set_params(kernel = 'rbf').fit(x,y)
print('prediction with rbf kernel')
print(clf.predict(xTest))


