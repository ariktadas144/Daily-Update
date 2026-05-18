#Pandas – Series, DataFrame, EDA, Data Cleaning, Data Manipulation

import pandas as pd

#Pandas- Series
series1=pd.Series([10,20,30,40,50])
print(series1)
#0    10
#1    20
#2    30
#3    40
#4    50
#dtype: int64

#Pandas- DataFrame
df=pd.DataFrame({"Name": ["Ram", "Shyam", "Mohan"], "Age": [25,30,35]})
print(df)
#    Name  Age
#0    Ram   25
#1  Shyam   30
#2  Mohan   35

#Exploratory Data Analysis (EDA)
print(df.head())
#    Name  Age
#0    Ram   25
#1  Shyam   30
#2  Mohan   35
print(df.tail())
#    Name  Age
#0    Ram   25
#1  Shyam   30
#2  Mohan   35
print(df.info())
print(df.describe())
#<class 'pandas.DataFrame'>
#RangeIndex: 3 entries, 0 to 2
#Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype
#---  ------  --------------  -----
# 0   Name    3 non-null      str  
# 1   Age     3 non-null      int64
#dtypes: int64(1), str(1)
#memory usage: 180.0 bytes

#Data Cleaning
df1= pd.DataFrame({
    "Name": ["Ram", "Shyam", "Mohan"],
    "Marks": [100, 90, None],
    "Age": [25, None, 30]
})
print(df1.isna()) 
print(df1.isna().count())
#        Age
#count   3.0
#mean   30.0
#std     5.0
#min    25.0
#25%    27.5
#50%    30.0
#75%    32.5
#max    35.0

#    Name  Marks    Age
#0  False  False  False
#1  False  False   True
#2  False   True  False
#Name     3
#Marks    3
#Age      3
#dtype: int64
df1.dropna(inplace=True)
df1["Age"].fillna(df1["Age"].mean(), inplace=True)
print(df1)
#Name  Marks   Age
#0  Ram  100.0  25.0

#Data Manipulation
df["age_after_5years"]=df["Age"]+5
print(df)
#    Name  Age  age_after_5years
#0    Ram   25                30
#1  Shyam   30                35
#2  Mohan   35                40