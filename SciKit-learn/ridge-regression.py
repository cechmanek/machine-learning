# ridge regression adds a weighting parameter to the cost function
# to penalize very large weights

# this is done to handle issues with colinearity between prediction
# parameters.

# cost = min(X*w-y)^2 + a*w^2

# X*w is our linear prediction
# y is our actual
# w is our weight
# a is our tunable weight parameter, a>=0

# larger a lead to more robustness against colinearity

from sklearn import linear_model

reg = linear_model.Ridge(alpha = 0.5)

#reg.fit([[0,0],[1,1],[2,2]], [0,1,2])

reg.fit([[0,0],[0,0],[1,1]], [0,0.1,1])

print(reg.coef_)
print(reg.intercept_)

print('--------------')

# RidgeCV() does cross validation of a (alpha parameter)
reg2 = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0])
reg2.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])  
print(reg2.alpha_)