# Smart City Traffic Stress Index - ML Regression Analysis

## Project Link
https://colab.research.google.com/drive/1sncNCZ7chzWtJ98lVKw7Rp9mlkRF5Gym?usp=sharing

## Project Overview

In this project I have performed an **End-to-End Exploratory Data Analysis (EDA)**, **Data Visualization**, and **Machine Learning Regression Modelling** on a Smart City Traffic Stress Dataset to uncover valuable insights regarding traffic conditions, driver behaviour, environmental factors, and their combined effect on a computed **stress index**.

The analysis includes the following steps:

- Data Loading & Inspection
- Missing Value & Duplicate Handling
- Statistical Analysis
- Univariate & Bivariate Analysis
- Skewness & Outlier Detection
- Correlation Analysis
- Variance Inflation Factor (VIF) Analysis
- Feature Engineering (One-Hot Encoding, Standard Scaling)
- Model Training & Evaluation (Linear Regression, Random Forest, XGBoost)
- Residual Analysis & Cross-Validation


# Dataset Information

The dataset contains smart-city traffic records capturing road conditions, driver characteristics, environmental factors, and a derived stress index for each observation.

## Sample Data (First 5 Rows)

| traffic_density | horn_events_per_min | avg_speed | signal_wait_time | weather_condition | road_quality_score | driver_experience_level | stress_index |
|---|---|---|---|---|---|---|---|
| 112 | 12.93 | 28.61 | 56.58 | Foggy | 7.31 | Intermediate | 68.37 |
| 61 | 7.43 | 54.22 | 35.64 | Rainy | 8.78 | Beginner | 47.14 |
| 102 | 11.07 | 41.42 | 54.61 | Clear | 8.34 | Intermediate | 55.02 |
| 24 | 1.54 | 69.86 | 16.09 | Clear | 6.29 | Expert | 22.71 |
| 116 | 11.60 | 33.01 | 62.51 | Clear | 8.19 | Expert | 49.91 |

## Feature Descriptions

| Column | Type | Description |
|---|---|---|
| `traffic_density` | Numerical | Number of vehicles on the road segment |
| `horn_events_per_min` | Numerical | Average horn honks recorded per minute |
| `avg_speed` | Numerical | Average speed of vehicles (km/h) |
| `signal_wait_time` | Numerical | Average waiting time at traffic signals (seconds) |
| `weather_condition` | Categorical | Prevailing weather — Clear, Rainy, Foggy, etc. |
| `road_quality_score` | Numerical | Road surface quality score (scale 0–10) |
| `driver_experience_level` | Categorical | Driver tier — Beginner, Intermediate, or Expert |
| `stress_index` | Numerical (Target) | Computed composite traffic stress score |


# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation & Cleaning |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| scikit-learn | ML Modelling, Scaling, Evaluation |
| XGBoost | Gradient Boosting Regression |
| Google Colab Notebook | Interactive Analysis |


# Steps Performed

## 1. Data Loading

Imported the dataset using Pandas:

```python
df = pd.read_csv('smart_city_traffic_stress_dataset.csv')
```

## 2. Initial Data Inspection

Performed the following to understand the dataset structure, inspect datatypes, and get a statistical overview:

```python
df.head()
df.dtypes
df.info()
df.describe(include="all")
```

## 3. Data Cleaning

### Missing Value Handling

Checked for null values across all columns:

```python
df.isnull().sum()
```

### Duplicate Detection

```python
df.duplicated().sum()
```

No duplicate records were found, so no removal was required.


# Exploratory Data Analysis (EDA)

## Univariate Analysis

Analyzed the distribution of individual columns using histograms and count plots.

### Histograms & KDE — Numerical Columns

```python
sns.histplot(df["traffic_density"], kde=True)
sns.histplot(df["horn_events_per_min"])
sns.histplot(df["avg_speed"])
sns.histplot(df["signal_wait_time"])
sns.histplot(df["road_quality_score"])
```

### Count Plots — Categorical Columns

```python
sns.countplot(x=df["weather_condition"])
sns.countplot(x=df["driver_experience_level"])
```

## Bivariate Analysis

Studied relationships between individual features and the target variable `stress_index`.

### Regression Plots — Numerical Features vs. Target

```python
sns.regplot(x=df["traffic_density"],      y=df["stress_index"], data=df, line_kws={"color": "red"})
sns.regplot(x=df["horn_events_per_min"],  y=df["stress_index"], data=df, line_kws={"color": "red"})
sns.regplot(x=df["avg_speed"],            y=df["stress_index"], data=df, line_kws={"color": "red"})
sns.regplot(x=df["road_quality_score"],   y=df["stress_index"], data=df, line_kws={"color": "red"})
```

### Box Plots — Categorical Features vs. Target

```python
sns.boxplot(x=df["weather_condition"],        y=df["stress_index"])
sns.boxplot(x=df["driver_experience_level"],  y=df["stress_index"])
```

## Skewness Analysis

Checked skewness of all numerical columns to understand distributional asymmetry:

```python
df.skew(numeric_only=True)
```

## Correlation Analysis

Generated a correlation heatmap to identify linear relationships between numerical features and the target:

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="rocket_r")
plt.show()
```

## Outlier Detection

Visualized outliers in the target variable using a box plot:

```python
sns.boxplot(df["stress_index"])
plt.show()
```

## Variance Inflation Factor (VIF) Analysis

Assessed multicollinearity among numerical predictors before modelling:

```python
X1 = df.drop(columns=["stress_index", "weather_condition", "driver_experience_level"])
vif_data = pd.DataFrame()
vif_data["Feature"] = X1.columns
```


# Model Building

## Feature Engineering

### One-Hot Encoding

Converted categorical columns (`weather_condition`, `driver_experience_level`) into binary dummy variables:

```python
df = pd.get_dummies(df, drop_first=True, dtype='int')
```

### Train-Test Split

```python
X = df.drop("stress_index", axis=1)
y = df["stress_index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Standard Scaling

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

## Linear Regression — Baseline Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Regression Metrics

```python
mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2   = r2_score(y_test, predictions)

print("MAE  = ", mae)
print("RMSE = ", rmse)
print("R2   = ", r2)
```

## Residual Analysis

Checked for homoscedasticity and normally distributed residuals:

```python
residuals = y_test - predictions

sns.scatterplot(x=predictions, y=residuals)
plt.axhline(0, color="red")
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.show()

sns.histplot(residuals, kde=True)
plt.show()
```

## Cross-Validation

```python
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(scores)
print("Mean CV R2:", scores.mean())
```

## Bias-Variance Check

```python
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

print("Training R2 score:", r2_score(y_train, y_train_pred))
print("Testing  R2 score:", r2_score(y_test,  y_test_pred))
```

## Multi-Model Comparison

Trained and evaluated three regression models and compared their performance:

```python
def evaluate_stress_regression_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression":        LinearRegression(),
        "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Regressor":        XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    performance_records = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "Model":            name,
            "MAE":              mean_absolute_error(y_test, predictions),
            "RMSE":             np.sqrt(mean_squared_error(y_test, predictions)),
            "R-squared (R2)":   r2_score(y_test, predictions)
        }
        performance_records.append(metrics)
    return pd.DataFrame(performance_records)

print(evaluate_stress_regression_models(X_train, X_test, y_train, y_test))
```


# Visualizations Performed

- Histograms with KDE overlays
- Count plots for categorical features
- Regression plots (feature vs. target)
- Box plots (categorical feature vs. target & outlier detection)
- Correlation heatmap
- Residual scatter plot
- Residual distribution histogram


# Key Insights

- Higher `traffic_density` and `signal_wait_time` show a strong positive correlation with `stress_index`
- `avg_speed` is negatively correlated with `stress_index` — faster-moving traffic produces lower stress readings
- Expert drivers tend to record lower stress indices compared to Beginner and Intermediate drivers
- Foggy and Rainy weather conditions are associated with elevated stress index values compared to Clear conditions
- `horn_events_per_min` is a meaningful predictor, suggesting acoustic congestion reflects overall traffic stress
- Ensemble models (Random Forest and XGBoost) outperform the Linear Regression baseline, capturing non-linear relationships between features and stress index