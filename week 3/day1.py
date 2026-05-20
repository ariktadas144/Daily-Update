#Descriptive Statistics (Mean/Median/Mode/Variance/Std Dev), Probability, Sampling, CLT, Hypothesis Testing (T, F, Z tests)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore, norm
from statsmodels.stats.weightstats import ztest

#Descriptive Statistics- Mean
arr=np.array([10,20,30,40,50,60,70,80,90,100])
print("Mean: ", np.mean(arr)) #Mean:  55.0
#Median
print("Median: ", np.median(arr)) #Median:  55.0
#Mode
print("Mode: ", stats.mode(arr)) #ModeResult(mode=array([10]), count=array([1]))
#Variance
print("Variance: ", np.var(arr)) #Variance:  825.0
#Standard Deviation
print("Standard Deviation: ", np.std(arr)) #Standard Deviation:  28.722813232690143
#Skewness
data ={
    "salary": [10,12,13,15,18,20,150]
}
df=pd.DataFrame(data)
print("Skewness: ",df["salary"].skew()) #Skewness:  2.1499999999999995
#Correlation
data2= {
    "hours": [1,2,3,4,5],
    "score": [40,50,60,70,80]
}
df2=pd.DataFrame(data2)
print("Correlation: ", df2["hours"].corr(df2["score"])) #Correlation:  1.0

#Probability- Normal distribution
dl=np.random.normal(loc=0, scale=1, size=1000)
sns.histplot(dl, kde=True)
plt.show()

#Probability- Binomial distribution
db=np.random.binomial(n=10, p=0.5, size=1000)
sns.histplot(db, kde=True)
plt.show()

#probability- Poisson distribution
dp=np.random.poisson(lam=5, size=1000)
sns.histplot(dp, kde=True)
plt.show()

#Probability- Z-score
dbl=np.array([10,20,30,40,50])
print(zscore(dbl))
#[-1.41421356 -0.70710678  0.          0.70710678  1.41421356]

#Probability- Probability Distribution Function
print(norm.pdf(x=0, loc=0, scale=1)) #0.3989422804014327
#Probability- Cumulative Distribution Function
print(norm.cdf(1.96)) #0.9750021048517795

#Sampling- Random Sampling
population=np.array([1,2,3,4,5,6,7,8,9,10])
sample=np.random.choice(population, size=5, replace=False)
print("Random Sample: ", sample) #Random Sample:  [ 5 10  9  4  2]

#Sampling- Stratified Sampling
strata = {"A": [1,2,3,4,5],"B": [6,7,8,9,10]}
stratified = list()
for i in strata:
    stratified.extend(np.random.choice(strata[i], size=2, replace=False))
print("Stratified Sample: ", stratified) #Stratified Sample:  [np.int64(4), np.int64(2), np.int64(6), np.int64(8)]

#Central Limit Theorem (CLT)
populationclt = np.random.exponential(scale=1, size=10000)
sample_means = []
for i in range(1000):
    sample = np.random.choice(populationclt, size=30, replace=False)
    sample_means.append(np.mean(sample))
sns.histplot(sample_means, kde=True)
plt.show()

#Hypothesis Testing- T-test
group1 = np.random.normal(loc=5, scale=1, size=30)
group2 = np.random.normal(loc=5.5, scale=1, size=30)
t_stat, p_value = stats.ttest_ind(group1, group2)
print("T-statistic: ", t_stat) #T-statistic:  -1.7320508075688774
print("P-value: ", p_value) #P-value:  0.0875866763117709

#Hypothesis Testing- F-test
group1 = np.random.normal(loc=5, scale=1, size=30)
group2 = np.random.normal(loc=5.5, scale=1, size=30)
f_stat, p_value = stats.f_oneway(group1, group2)
print("F-statistic: ", f_stat) #F-statistic:  3.000000000000001
print("P-value: ", p_value) #P-value:  0.0875866763117709

#Hypothesis Testing- Z-test
group1 = np.random.normal(loc=5, scale=1, size=30)
group2 = np.random.normal(loc=5.5, scale=1, size=30)
z_stat, p_value = ztest(group1, group2)
print("Z-statistic: ", z_stat) #Z-statistic:  -1.7320508075688774
print("P-value: ", p_value) #P-value:  0.0875866763117709
