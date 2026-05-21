#Simple & Multiple Linear Regression, Assumptions, Evaluation Metrics (R², RMSE), Regularisation (Ridge, Lasso), Feature Scaling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Simple Linear Regression
hours = np.array([1,2,3,4,5]).reshape(-1,1)
scores=np.array([40,50,60, 70, 80])
model=LinearRegression()
model.fit(hours, scores)
predictions=model.predict(hours)
print(model.coef_)
print(model.intercept_)
plt.scatter(hours, scores)
plt.plot(hours, predictions)
plt.xlabel('Hours studied')
plt.ylabel('Score')
plt.show()
#[10.]
#30.000000000000007

#Multiple Linear Regression
data = {
    "experience": [1, 2, 3, 5, 6, 8, 10, 12, 15, 20],
    "projects": [2, 3, 2, 5, 4, 7, 6, 8, 11, 10],
    "salary": [51025, 54519, 64327, 78801, 76118, 93551, 102602, 114674, 131751, 151525]
}
df=pd.DataFrame(data)
X=df[["experience", "projects"]]
y=df["salary"]
model2=LinearRegression()
model2.fit(X,y)
predictions1 = model2.predict(X)
print(model2.coef_)
print(model2.intercept_)
#[4512.85545024 1895.38388626]
#43890.65876777252

#Assumptions
residuals=y-predictions1
sns.scatterplot(x=predictions1, y=residuals, data=df)
plt.show()

#Evaluation metrics- R², RMSE
mse=mean_squared_error(y, predictions1)
rmse=np.sqrt(mse)
r2=r2_score(y, predictions1)
print("RMSE score: ", rmse)
print("R2 score: ", r2)
#RMSE:  2278.3551047657893
#R2 score:  0.9948236878675251

#Feature Scaling- Standard Scaling
scaler=StandardScaler()
X_scaled= scaler.fit_transform(X)
print(X_scaled)
#array([[-1.24285802, -1.25555534],
#       [-1.07023885, -0.92514604],
#       [-0.89761968, -1.25555534],
#       [-0.55238134, -0.26432744],
#       [-0.37976217, -0.59473674],
#       [-0.03452383,  0.39649116],
#       [ 0.31071451,  0.06608186],
#       [ 0.65595285,  0.72690046],
#       [ 1.17381036,  1.71812836],
#       [ 2.03690621,  1.38771906]])

#Feature Scaling- MinMax Scaling
scaler2=MinMaxScaler()
X_scaled_2=scaler2.fit_transform(X)
print(X_scaled_2)
#array([[0.        , 0.        ],
#       [0.05263158, 0.11111111],
#       [0.10526316, 0.        ],
#       [0.21052632, 0.33333333],
#       [0.26315789, 0.22222222],
#       [0.36842105, 0.55555556],
#       [0.47368421, 0.44444444],
#       [0.57894737, 0.66666667],
#       [0.73684211, 1.        ],
#       [1.        , 0.88888889]])

#Regularization- Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled,y)
print(ridge.coef_)
#[ 19076.09290519 11239.26136815]

#Regularization- Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_scaled, y)
print(lasso.coef_)
#[26159.2919331   5720.59011561]
