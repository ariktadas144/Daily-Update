import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

#Creation of stress_level column
bins = [0,30,60,110]
labels = ["Low", "Medium", "High"]
df["stress_level"]= pd.cut(df["stress_index"], bins=bins, labels=labels)
df[["stress_index", "stress_level"]].head()

#Null analysis of the new stress_level column
df["stress_level"].isnull().sum()
df["stress_level"]= df["stress_level"].fillna(df["stress_level"].mode()[0])
df["stress_level"].isnull().sum()

sns.countplot(x="stress_level", data=df)
plt.show()

#Split the data into features and target variable
X = df.drop(columns=["stress_index", "stress_level"])
y1 = df["stress_level"]

#Label Encoding 
le = LabelEncoder()
y_df = pd.DataFrame(le.fit_transform(y1), columns=['stress_level'], index=y1.index)
y = y_df['stress_level']

#One-Hot Encoding
X = pd.get_dummies(X,drop_first=True,dtype='int')

#Train and test data split
X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2,random_state=42))

#Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model training and metric evaluation
def evaluate_classification_models(X_train, X_test, y_train, y_test):
  models = {
        "Logistic Regression": LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
    }
  performance_records = []
  for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, predictions, target_names=labels))
    metrics = {
        "Model": name,
        "Overall Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision_score(y_test, predictions, average='macro'),
        "Macro Recall": recall_score(y_test, predictions, average='macro'),
        "Macro F1-Score": f1_score(y_test, predictions, average='macro')
        }
    performance_records.append(metrics)
    print(f"[Confusion Matrix for {name}]")
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(
            cm, 
            index=[f"Actual {lbl}" for lbl in labels],
            columns=[f"Predicted {lbl}" for lbl in labels]
        )
    print(cm_df)
    print("\n" + " "*20)
  results_df = pd.DataFrame(performance_records)
  results_df = results_df.sort_values(by="Overall Accuracy", ascending=False).reset_index(drop=True)
  return results_df

print(evaluate_classification_models(X_train, X_test, y_train, y_test))
