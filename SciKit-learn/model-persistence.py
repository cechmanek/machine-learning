# model persistence

# can save models you've built using pickle library

from sklearn import svm
from sklearn import datasets

# initialize predictor object
clf = svm.SVC()

iris = datasets.load_iris()
x, y = iris.data, iris.target

# train predictor using all data
clf.fit(x, y)

# now save the model
import pickle
s = pickle.dumps(clf)
# save to a string, only active this python session

# can now load trained predictor model
clf2 = pickle.loads(s)

myPrediction = clf2.predict(x[0:1])
# same as x[0], but python doesn't like 1d arrays like this
print('my prediction is: ',myPrediction)

print('the answer is:', y[0])


# instead of pickle you can use joblib
from sklearn.externals import joblib

# can only save to disk, not temporary string
joblib.dump(clf, 'digitsModel.pkl')

clf3 = joblib.load('digitsModel.pkl')

