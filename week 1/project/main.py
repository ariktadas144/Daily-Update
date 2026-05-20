import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading dataset
df=pd.read_csv('retail_store_sales.csv')

#Basic Inspection
df.head()
df.shape
df.dtypes
df.info()
df.describe()

#Null Data Analysis and Handling
df.isna().sum()
df["Item"]=df["Item"].fillna(df["Item"].mode()[0])
df["Price Per Unit"]=df["Price Per Unit"].fillna(df["Price Per Unit"].mean())
df["Quantity"]=df["Quantity"].fillna(df["Quantity"].mean())
df["Total Spent"]=df["Total Spent"].fillna(df["Total Spent"].mean())
df["Discount Applied"]=df["Discount Applied"].fillna(df["Discount Applied"].mode()[0])
df["Discount Applied"] = df["Discount Applied"].astype(bool)
df.isna().sum() #Rechecking if all null values are handled

#Checking for Duplicate Values
df.duplicated().sum()

#Getting a Statistical Summary of the Data
df.describe()

#Univariate Analysis of the Numerical Columns
sns.histplot(df["Price Per Unit"], kde=True)
plt.show()
sns.histplot(df["Quantity"], kde=True)
plt.show()
sns.histplot(df["Total Spent"], kde=True)
plt.show()

#Univariate Analysis of the Categorical Columns
sns.countplot(x=df["Category"])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
sns.countplot(x=df["Payment Method"])
plt.show()
sns.countplot(x=df["Location"])
plt.show()
sns.countplot(x=df["Discount Applied"])
plt.show()

#Bivariate Analysis of Numerical v/s Numerical Columns
sns.regplot(x="Price Per Unit", y="Total Spent", data=df)
plt.show()

#Bivariate Analysis of Categorical v/s Numerical Columns
sns.boxplot(x="Category", y="Total Spent", data=df)
plt.xticks(rotation=45, ha='right')
plt.show()
sns.boxplot(x="Payment Method", y="Total Spent", data=df)
plt.show()
sns.boxplot(x="Location", y="Total Spent", data=df)
plt.show()

#Bivariate Analysis of Categorical v/s Categorical Columns
sns.countplot(x="Category", hue="Payment Method", data=df)
plt.xticks(rotation=45, ha='right')
plt.show()
sns.countplot(x="Category", hue="Location", data=df)
plt.xticks(rotation=45, ha='right')
plt.show()

#Correlation Analsis using Heatmap
corr=df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="rocket_r")
plt.show()
