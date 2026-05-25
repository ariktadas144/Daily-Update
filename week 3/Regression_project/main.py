import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#Loading the Dataset
df=pd.read_csv('smart_city_traffic_stress_dataset.csv')

#Inspecting the Dataset
df.head()
df.dtypes
df.info()
df.describe(include="all")

#Checking for Null and Duplicate Values
df.isnull().sum()
df.duplicated().sum()

#Univariate Analysis of Numerical Columns
sns.histplot(df["traffic_density"],kde=True)
plt.show()
sns.histplot(df["horn_events_per_min"])
plt.show()
sns.histplot(df["avg_speed"])
plt.show()
sns.histplot(df["signal_wait_time"])
plt.show()
sns.countplot(x=df["weather_condition"])
plt.show()
sns.histplot(df["road_quality_score"])
plt.show()
sns.countplot(x=df["driver_experience_level"])
plt.show()

#Bivariate Analysis of Featues v/s Target Variable (stress_index)
sns.regplot(x=df["traffic_density"], y=df["stress_index"], data=df, line_kws={"color": "red"})
plt.show()
sns.regplot(x=df["horn_events_per_min"], y=df["stress_index"], data=df, line_kws={"color": "red"})
plt.show()
sns.regplot(x=df["avg_speed"], y=df["stress_index"], data=df, line_kws={"color": "red"})
plt.show()
sns.regplot(x=df["road_quality_score"], y=df["stress_index"], data=df, line_kws={"color": "red"})
plt.show()
sns.boxplot(x=df["weather_condition"], y=df["stress_index"])
plt.show()
sns.boxplot(x=df["driver_experience_level"], y=df["stress_index"])
plt.show()

#Checking for Skewness
df.skew(numeric_only=True)

#Correlation Analysis
corr = df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap="rocket_r")
plt.show()

#Outlier Detection using Box Plots
sns.boxplot(df["stress_index"])
plt.show()

#Variance inflation factor analysis
X1 = df.drop(columns=["stress_index", "weather_condition", "driver_experience_level"])
vif_data = pd.DataFrame()
vif_data["Feature"] = X1.columns

#One-Hot Encoding
df = pd.get_dummies(df,drop_first=True,dtype='int')

#Splitting the data into features and target
X = df.drop("stress_index", axis=1)
y = df["stress_index"]

#Splitting it into train and test feature dataset
X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2,random_state=42))

#Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Linear Regression model training
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

#Regression Metrics
mae = mean_absolute_error(y_test,predictions)
rmse = np.sqrt(mean_squared_error( y_test,predictions))
r2 = r2_score(y_test, predictions)

print("MSE = ",mae)
print("RMSE = ",rmse)
print("R2 score = ",r2)

#Residual errors 
residuals = y_test - predictions

sns.scatterplot(x=predictions,y=residuals)
plt.axhline(0,color="red")
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.show()
sns.histplot(residuals,kde=True) #CChecking for homodescacity
plt.show()

#Cross Validation Score Calculation
scores = cross_val_score(model,X,y,cv=5,scoring="r2")
print(scores)
print(scores.mean())

#Checking the variance in results
y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)
print("Training R2 score: ", r2_score(y_train, y_train_predictions))
print("Testing R2 score: ", r2_score(y_test, y_test_predictions))

#Regression ml model training
def evaluate_stress_regression_models(X_train, X_test, y_train, y_test):
  models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(objective='reg:squarederror', random_state=42)
    }
  performance_records = []
  for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  
    r2 = r2_score(y_test, predictions)
    metrics = {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R-squared (R2)": r2
    }
    performance_records.append(metrics)
  results_df = pd.DataFrame(performance_records)
  return results_df

print(evaluate_stress_regression_models(X_train, X_test, y_train, y_test))
