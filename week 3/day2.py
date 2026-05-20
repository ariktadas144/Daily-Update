#ANOVA, Chi-Square; Correlation vs Causation; Intro to GitHub (Repo, Commit, Push)

import numpy as np
from scipy import stats

#Anova
group1=np.random.normal(loc=5, scale=1, size=100)
group2=np.random.normal(loc=5.5, scale=1, size=100)
group3=np.random.normal(loc=6, scale=1, size=100)
f_stats, p_value = stats.f_oneway(group1, group2, group3)
print("F-statistic: ", f_stats) #F-statistic:  39.96565451219911
print("P-value: ", p_value) #P-value:  4.2598483028270506e-16

#Chi-Square Test (contingency table)
observed=np.array([[5,10], [20,30]])
chi, p_value, dof, expected = stats.chi2_contingency(observed)
print("Chi-Square statistic: ", chi)
print("P-value: ", p_value)
print("Degrees of freedom: ", dof)
print("Expected frequencies:\n", expected)
#Chi-Square statistic:  0.026541666666666745
#P-value:  0.8705844675726829
#Degrees of freedom:  1
#Expected frequencies:
# [[ 5.76923077  9.23076923]
# [19.23076923 30.76923077]]

#Correlation vs Causation
#Correlation checks if two variables move together or not
#If both increase, positive correlation; if one increases and other decreases, negative correlation; if it is zero, no linear relationship
a = np.random.rand(100)
b = np.random.rand(100)
corr = np.corrcoef(a, b)[0, 1]
print("Correlation: ", corr)
#Correlation:  0.21997709477550142

#Causation shows how one variable directly causes change in another variable

#Correlation v/s Causattion: Sometimes variables can correlate purely coincidentally known as spurious correlation
#So, correlation does not imply causation, but vice versa is true. 
