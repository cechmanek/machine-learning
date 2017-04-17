# multiclass and multilabel fitting
# multiclass classifiers

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

x = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator = SVC(random_state=0))
myPredict = classif.fit(x,y).predict(x)

print(myPredict)

'''
classifier is fit on 1D array of multiclass labels
so .predict() provides multiclass predictions.
can also fit on 2D array 
'''
y = LabelBinarizer().fit_transform(y)
print('2d prediction')
print(classif.fit(x,y).predict(x))
# prints 2d array of corresponding multilabel predictions
# row of zeros indicates no match was found at all


# can also have multiple matches
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.preprocessing

y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = sklearn.preprocessing.MultiLabelBinarizer().fit_transform(y)
multiClass = classif.fit(x,y).predict(x)
print('multiple class matching')
print(multiClass)





