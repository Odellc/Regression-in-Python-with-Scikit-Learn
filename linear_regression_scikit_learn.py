import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1,1))

y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression().fit(x,y)

print(model.score(x,y)) # R^2 value of the linear model

print(model.intercept_) # print intercept

print('slope =', model.coef_)

#predict y based on x
y_pred = model.predict(x)

print("predicted response:", y_pred, sep = "\n")

x_new = np.array([6, 10, 33, 28]).reshape(-1,1)

y_new = model.predict(x_new)

print(y_new) # do prediction off new x values

## Multiple Linear Regression

xm = [[0,1], [5,1], [15,2], [25,5],[35,11], [45,15], [55,34], [60,35]]

ym = [4, 5, 20, 14, 32, 22, 38, 43]

xm,ym = np.array(xm), np.array(ym)

print(xm)

model2 = LinearRegression().fit(xm,ym)

r_sq = model2.score(xm, ym)
print(r_sq)

ym_pred = model2.predict(xm)

print("model 2 prediction on training data:", ym_pred)

xm_new = [[2,10], [20,15]]

ym_pred_new = model2.predict(xm_new)

print("model 2 prediction on test data:", ym_pred_new)

# Polynomial Multivarate Regression

transformer = PolynomialFeatures(degree=2, include_bias=False)

transformer.fit(x)
#print(transformer)

x_ = transformer.transform(x)

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

print(x_)

model = LinearRegression().fit(x_,y)

r_sq_new = model.score(x_,y)
print(r_sq_new)

