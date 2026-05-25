# Smart City Traffic Stress Level - ML Classification Analysis

## Project Link
https://colab.research.google.com/drive/1xGzxZv3t3Dq9kFS-2N4cJwjim47WvK_Q?usp=sharing

## Project Overview

In this project I have performed an **End-to-End Exploratory Data Analysis (EDA)**, **Data Visualization**, and **Machine Learning Multi-Class Classification** on a Smart City Traffic Stress Dataset to classify traffic conditions into discrete stress levels — **Low**, **Medium**, and **High** — based on road conditions, driver characteristics, and environmental factors.

The analysis includes the following steps:

- Data Loading & Inspection
- Missing Value & Duplicate Handling
- Statistical Analysis
- Univariate & Bivariate Analysis
- Skewness & Outlier Detection
- Correlation Analysis
- Variance Inflation Factor (VIF) Analysis
- Target Variable Engineering (Binning `stress_index` into `stress_level`)
- Feature Engineering (Label Encoding, One-Hot Encoding, Standard Scaling)
- Multi-Class Model Training & Evaluation (Logistic Regression, Random Forest, SVM, XGBoost)
- Classification Reports & Confusion Matrices


# Dataset Information

The dataset contains smart-city traffic records capturing road conditions, driver characteristics, and environmental factors. The continuous `stress_index` column is binned into a categorical `stress_level` target for classification.

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
| `stress_index` | Numerical (Source) | Raw composite traffic stress score used to derive the target |
| `stress_level` | Categorical (Target) | Binned stress class — Low, Medium, or High |

## Target Variable — Stress Level Bins

| Class | stress_index Range |
|---|---|
| Low | 0 – 30 |
| Medium | 30 – 60 |
| High | 60 – 110 |


# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation & Cleaning |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| scikit-learn | ML Modelling, Encoding, Scaling, Evaluation |
| XGBoost | Gradient Boosting Classification |
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

Studied relationships between individual features and the continuous `stress_index` column.

### Regression Plots — Numerical Features vs. stress_index

```python
sns.regplot(x=df["traffic_density"],      y=df["stress_index"], data=df, line_kws={"color": "red"})
sns.regplot(x=df["horn_events_per_min"],  y=df["stress_index"], data=df, line_kws={"color": "red"})
sns.regplot(x=df["avg_speed"],            y=df["stress_index"], data=df, line_kws={"color": "red"})
sns.regplot(x=df["road_quality_score"],   y=df["stress_index"], data=df, line_kws={"color": "red"})
```

### Box Plots — Categorical Features vs. stress_index

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

Generated a correlation heatmap to identify linear relationships between numerical features:

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="rocket_r")
plt.show()
```

## Outlier Detection

Visualized outliers in the source variable using a box plot:

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

# Target Variable Engineering

## Binning stress_index into stress_level

The continuous `stress_index` column was discretized into three ordinal classes using `pd.cut()`:

```python
bins   = [0, 30, 60, 110]
labels = ["Low", "Medium", "High"]

df["stress_level"] = pd.cut(df["stress_index"], bins=bins, labels=labels)
df[["stress_index", "stress_level"]].head()
```

## Null Handling for stress_level

Boundary values outside the defined bins can produce nulls, which were imputed using the mode:

```python
df["stress_level"].isnull().sum()
df["stress_level"] = df["stress_level"].fillna(df["stress_level"].mode()[0])
df["stress_level"].isnull().sum()
```

## Class Distribution

Visualized the balance of the newly created target classes:

```python
sns.countplot(x="stress_level", data=df)
plt.show()
```

# Model Building

## Feature & Target Split

```python
X  = df.drop(columns=["stress_index", "stress_level"])
y1 = df["stress_level"]
```

## Feature Engineering

### Label Encoding — Target Variable

Converted the ordinal string labels into integer class indices:

```python
le = LabelEncoder()
y_df = pd.DataFrame(le.fit_transform(y1), columns=['stress_level'], index=y1.index)
y = y_df['stress_level']
```

### One-Hot Encoding — Categorical Features

Converted `weather_condition` and `driver_experience_level` into binary dummy variables:

```python
X = pd.get_dummies(X, drop_first=True, dtype='int')
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Standard Scaling

```python
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

## Multi-Model Classification

Trained and evaluated four classification models, printing a full classification report and confusion matrix for each:

```python
def evaluate_classification_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression":    LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42),
        "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "XGBoost":                XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
    }
    performance_records = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, predictions, target_names=labels))

        metrics = {
            "Model":            name,
            "Overall Accuracy": accuracy_score(y_test, predictions),
            "Macro Precision":  precision_score(y_test, predictions, average='macro'),
            "Macro Recall":     recall_score(y_test, predictions, average='macro'),
            "Macro F1-Score":   f1_score(y_test, predictions, average='macro')
        }
        performance_records.append(metrics)

        cm = confusion_matrix(y_test, predictions)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {lbl}"    for lbl in labels],
            columns=[f"Predicted {lbl}" for lbl in labels]
        )
        print(f"[Confusion Matrix for {name}]")
        print(cm_df)

    results_df = pd.DataFrame(performance_records)
    results_df = results_df.sort_values(by="Overall Accuracy", ascending=False).reset_index(drop=True)
    return results_df

print(evaluate_classification_models(X_train, X_test, y_train, y_test))
```

## Evaluation Metrics Used

| Metric | Description |
|---|---|
| Overall Accuracy | Proportion of correctly classified samples |
| Macro Precision | Average precision across all three classes (unweighted) |
| Macro Recall | Average recall across all three classes (unweighted) |
| Macro F1-Score | Harmonic mean of Macro Precision and Macro Recall |
| Classification Report | Per-class precision, recall, F1, and support |
| Confusion Matrix | Actual vs. predicted class counts for each model |


# Visualizations Performed

- Histograms with KDE overlays
- Count plots for categorical features and target class distribution
- Regression plots (numerical feature vs. stress_index)
- Box plots (categorical feature vs. stress_index & outlier detection)
- Correlation heatmap
- Confusion matrices per model


# Key Insights

- Binning `stress_index` into Low / Medium / High transforms the problem into a well-structured multi-class classification task
- Higher `traffic_density` and `signal_wait_time` are strong predictors of the High stress class
- Expert drivers are more frequently associated with the Low stress class, confirming the importance of `driver_experience_level` as a feature
- Foggy and Rainy weather conditions contribute disproportionately to Medium and High stress classifications
- Ensemble models (Random Forest and XGBoost) are expected to outperform Logistic Regression and SVM by capturing complex non-linear decision boundaries between stress classes
- The confusion matrix highlights that the Medium class is the hardest to distinguish, as its index range overlaps in behaviour with both Low and High boundary cases