# Retail Sales Data Analysis & Visualization

## Project Link
https://colab.research.google.com/drive/195Ad7uWy-_xasG5su2W7t6VTW-uEMSJN?usp=sharing

## Project Overview

In this project I have performed an **End-to-End Exploratory Data Analysis (EDA)** and **Data Visualization** on a Retail Sales Dataset to uncover valuable insights regarding the products purchased, the amount spend, the quantity of each product purchased, the mode of payments and overall customer behaviour.

The analysis includes the following steps:

- Data Cleaning
- Missing Value Handling
- Duplicate Detection
- Statistical Analysis
- Univariate & Bivariate Analysis
- Correlation Analysis
- Data Visualization using Matplotlib & Seaborn

# Dataset Information

The dataset contains retail transaction records with customer purchases, payment methods, discounts, and transaction details.

## Sample Data (The first 5 rows)

| Transaction ID | Customer ID | Category | Item | Price Per Unit | Quantity | Total Spent | Payment Method | Location | Transaction Date | Discount Applied |
|---|---|---|---|---|---|---|---|---|---|---|
| TXN_6867343 | CUST_09 | Patisserie | Item_10_PAT | 18.5 | 10 | 185.0 | Digital Wallet | Online | 2024-04-08 | True |
| TXN_3731986 | CUST_22 | Milk Products | Item_17_MILK | 29.0 | 9 | 261.0 | Digital Wallet | Online | 2023-07-23 | True |
| TXN_9303719 | CUST_02 | Butchers | Item_12_BUT | 21.5 | 2 | 43.0 | Credit Card | Online | 2022-10-05 | False |
| TXN_9458126 | CUST_06 | Beverages | Item_16_BEV | 27.5 | 9 | 247.5 | Credit Card | Online | 2022-05-07 | NaN |
| TXN_4575373 | CUST_05 | Food | Item_6_FOOD | 12.5 | 7 | 87.5 | Digital Wallet | Online | 2022-10-02 | False |

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation & Cleaning |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Google Colab Notebook | Interactive Analysis |

# Steps Performed in EDA

## 1️. Data Loading

- Imported dataset using Pandas

```python
df = pd.read_csv("retail_sales.csv")
```

## 2️. Initial Data Inspection

Performed:

- `df.head()`
- `df.shape`
- `df.dtypes`
- `df.info()`
- `df.describe()`

Purpose:
- Understand dataset structure
- Inspect datatypes
- Get an overview of the dataset 

## 3️. Data Cleaning

### Missing Value Handling

- Checked null values using:

```python
df.isna().sum()
```

- Filled or handled missing values appropriately
  - For numerical columns having null values, used the mean of the column to fill in the missing values like:
    ```python
    df["Price Per Unit"]=df["Price Per Unit"].fillna(df["Price Per Unit"].mean())
    ```
  - For categorical columns having null values, used the mode of the column to fill in the missing values like:
    ```python
    df["Item"]=df["Item"].fillna(df["Item"].mode()[0])
    ```

### Duplicate Detection

```python
df.duplicated().sum()
```

Removed duplicate records if present (In this case, there were no duplicate values in the dataset so didn't require duplicates handling)

### Statistical Summary

Displayed an overview of the data after cleaning mentioing its mean, standard deviation, minimum value, maximum value and quartile values

```python
df.describe()
```

# Exploratory Data Analysis (EDA)

## Univariate Analysis

Analyzed individual columns using:

### Histograms and KDE
- For numerical datas like "Price Per Unit", "Quantity" and "Total Spent"
  ```python
  sns.histplot(df["Price Per Unit"], kde=True)
  ```
  
### Counter plots
- For categorical data like "Category", "Payment Method", "Location" and "Discount Applied"
  ```python
  sns.countplot(x=df["Category"])
  ```

## Bivariate Analysis

Studied relationships between variables using:

### Regression Plots
- For numerical v/s numerical data like "Price Per Unit" v/s "Total Spent"
  ```python
  sns.regplot(x="Price Per Unit", y="Total Spent", data=df)
  ```
  
### Boxplots
- For categorical v/s numerical data like "Category" v/s "Total Spent"
  ```python
  sns.boxplot(x="Category", y="Total Spent", data=df)
  ```

### Counter plots
- For categorical v/s categorical data like "Category" v/s "Payment Method"
  ```python
  sns.countplot(x="Category", hue="Payment Method", data=df)
  ```

## Correlation Analysis

Generated correlation heatmap to identify relationships between numerical features.

```python
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="rocket_r")
```

# Visualizations Performed

The project includes:

- Histograms
- Boxplots
- Counter plots
- Regression plots
- Heatmaps

# Key Insights

Some insights discovered from the analysis:

- Very few people have spent more than 150 and majority's total spent amount is less than 150 
- There has been almost equal comparable sales in all the categories
- The payment methods used and the location of purchase (in-store and online) are also almost comparable and equal
- A lot of discounts have been applied to the purchases
- All the columns contain atleast a few outliers
