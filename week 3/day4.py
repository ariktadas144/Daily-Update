#Logistic Regression, Decision Trees, Random Forest; Confusion Matrix, ROC-AUC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Logistic Regression (Clasification)
data = {
    "hours_studied":[3,5,7,9,1,2],
    "passed": [0,1,1,1,0,0]
}
df=pd.DataFrame(data)
X=[["hours_studied"]]
Y=["passed"]
model=LogisticRegression()
model.fit(X,Y)
predictions=model.predict(X)
print(predictions) #[0 1 1 1 0 0]
probabilities=model.predict_proba(X)
print(probabilities) 
#[[0.77116173 0.22883827]
# [0.30198353 0.69801647]
# [0.05261905 0.94738095]
# [0.00708    0.99292   ]
# [0.96330167 0.03669833]
# [0.90389389 0.09610611]]

#Classification Evaluation Metrics
print(accuracy_score(Y,predictions)) #1.0
print(recall_score(Y, predictions)) #1.0
print(precision_score(Y, predictions)) #1.0
print(f1_score(Y, predictions)) #1.0
print(confusion_matrix(Y, predictions))
#[[3 0]
# [0 3]]

#Decision Tree
model2=DecisionTreeClassifier(max_depth=3)
model2.fit(X, Y)
predictions2=model2.predict(X)
print(predictions2) #[0 1 1 1 0 0]

#Random Forest
model3=RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X, Y)
predictions3=model3.predict(X)
print(predictions3) #[0 1 1 1 0 0]

#ROC-AUC
print(roc_auc_score(Y, probabilities[:, 1])) #1.0
print(roc_auc_score(Y, model2.predict_proba(X)[:, 1])) #1.0
print(roc_auc_score(Y, model3.predict_proba(X)[:, 1])) #1.0
