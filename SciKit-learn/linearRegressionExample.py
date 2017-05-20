
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# load diabetes dataset

diabetes = datasets.load_diabetes()

# use only the first feature for this example
diabetes_x = diabetes.data[:,np.newaxis,2]

# split data into training and testing
diabetes_x_train = diabetes_x[:-20] # train on all but last 20
diabetes_x_test = diabetes_x[-20:] # test on last 20

# split target into training and testing
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# create a linear reg object
regr = linear_model.LinearRegression()

# train the model object
regr.fit(diabetes_x_train,diabetes_y_train)

# inspect the results
print('Coefficients: \n', regr.coef_)

# mean squared error
mse = np.mean( (regr.predict(diabetes_x_test)-diabetes_y_test)**2 )
print('Mean Squared Error: %.2f' %mse)

# variance score. 1 is a perfect score
varScore = regr.score(diabetes_x_test, diabetes_y_test)
print('Variance: \n %.2f' %varScore )

# plot everything

plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, regr.predict(diabetes_x_test),color='blue')
plt.show()