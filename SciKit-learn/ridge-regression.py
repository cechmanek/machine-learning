from sklearn import linear_model

reg = linear_model.Ridge(alpha = 0.5)

#reg.fit([[0,0],[1,1],[2,2]], [0,1,2])

reg.fit([[0,0],[0,0],[1,1]], [0,0.1,1])

print(reg.coef_)
print(reg.intercept_)


print('--------------')

reg2 = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0])
reg2.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])  
print(reg2.alpha_)