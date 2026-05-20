#Pandas – GroupBy, Aggregation, Merge, Concat; Matplotlib & Seaborn visualisation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas- GroupBy
data= {
    "Name": ["Ram", "Mohan", "Rohan", "Shyam", "Mohan", "Shyam", "Rohan", "Ram", "Shyam", "Ram", "Rohan", "Mohan"],
    "Age": [25, 30, 27, 32, 30, 32, 27, 25, 32, 25, 27, 30],
    "Marks": [90,95,75,99,68,80, 84, 93, 89, 78, 82, 95]
}
df=pd.DataFrame(data)
print(df.groupby("Name")["Marks"].mean().reset_index())
print(df.groupby(["Name", "Age"])["Marks"].mean().reset_index())
#    Name      Marks
#0  Mohan  86.000000
#1    Ram  87.000000
#2  Rohan  80.333333
#3  Shyam  89.333333
#    Name  Age      Marks
#0  Mohan   30  86.000000
#1    Ram   25  87.000000
#2  Rohan   27  80.333333
#3  Shyam   32  89.333333

#Pandas- Aggregation
print(df.groupby("Name")["Marks"].agg(["mean", "max", "min"]))
#            mean  max  min
#Name                      
#Mohan  86.000000   95   68
#Ram    87.000000   93   78
#Rohan  80.333333   84   75
#Shyam  89.333333   99   80

#Pandas- Merge
names= pd.DataFrame({
    "id": [1,2,3,4],
    "name": ["Ram", "Mohan", "Rohan", "Shyam"]
})
ages= pd.DataFrame({
    "id": [1,2,3,4],
    "age": [25,30,27,32]
})
print(pd.merge(names, ages, on="id"))
print(pd.merge(names, ages, on="id", how="inner"))
print(pd.merge(names, ages, on="id", how="left"))
print(pd.merge(names, ages, on="id", how="right"))
print(pd.merge(names, ages, on="id", how="outer"))
#   id   name  age
#0   1    Ram   25
#1   2  Mohan   30
#2   3  Rohan   27
#3   4  Shyam   32
#   id   name  age
#0   1    Ram   25
#1   2  Mohan   30
#2   3  Rohan   27
#3   4  Shyam   32
#   id   name  age
#0   1    Ram   25
#1   2  Mohan   30
#2   3  Rohan   27
#3   4  Shyam   32
#   id   name  age
#0   1    Ram   25
#1   2  Mohan   30
#2   3  Rohan   27
#3   4  Shyam   32

#Pandas- Concat
print(pd.concat([names, ages], axis=1))
#   id   name  id  age
#0   1    Ram   1   25
#1   2  Mohan   2   30
#2   3  Rohan   3   27
#3   4  Shyam   4   32

#Matplotlib - 2D Line Plot
data1 = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "Fiction_Sales": [4500, 4200, 5100, 4800, 5300, 6200, 6500, 5900, 5400, 5800, 7100, 9500],
    "NonFiction_Sales": [3200, 3100, 3800, 4100, 3900, 3500, 3600, 4000, 4500, 4800, 5500, 7800],
    "Children_Sales": [1800, 1900, 2200, 2500, 2800, 3400, 4100, 3900, 2700, 2900, 4200, 6800],
    "Store_Visits": [12000, 11500, 14000, 14500, 15000, 17500, 18000, 16800, 15200, 16000, 21000, 28500],
    "Promo_Active": [False, False, True, False, True, True, True, False, False, True, True, True]
}
df2=pd.DataFrame(data1)
plt.style.use("classic")
plt.plot(df2["Month"], df2["Fiction_Sales"], color="blue", linestyle="solid", linewidth=3, marker='D', markersize=10, label="Fiction")
plt.plot(df2["Month"], df2["NonFiction_Sales"], color="black", linestyle="solid", linewidth=3, marker='o', markersize=10, label="Non-Fiction")
plt.title("Fiction v/s Non-Fiction sales per month")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.ylim(0, 10000)
plt.grid()
plt.scatter(df2["Month"], df2["Store_Visits"])
plt.show()

#Matplotlib- Scatter Plot
plt.scatter(df2["Month"], df2["Children_Sales"])
plt.show()

#Matplotlib- Bar plot
names = ["A","B","C"]
scores = [80,90,70]
plt.bar(names, scores)
plt.show()

#Matplotlib- Histogram
marks = [50,60,70,80,90,90,95]
plt.hist(marks)
plt.show()

#Seaborn 
df=sns.load_dataset("tips")
sns.set_style("whitegrid")
#Counter plot
sns.countplot(x="day", data=df)
plt.show()
#Histogram
sns.histplot(df["total_bill"])
plt.show()
#Scatter plot
sns.scatterplot(x="total_bill", y="tip", data=df)
plt.show()
#Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="rocket_r", annot=True)
plt.show()
#Box plot
sns.boxplot(x="day", y="total_bill", data=df)
plt.show()
#Regression plot
sns.regplot(x="total_bill", y="tip", data=df)
plt.show()
