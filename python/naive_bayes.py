# Rafael Henriques
# 15-Jul-2019

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # plotting

# Read Dataset
print('Read Dataset')
df = pd.read_csv("../dataset/iris.data")

print(df.shape)
print(df.info())
print(df.head(4));

print('Pre-process')
# Prepare the training and testing data
# 4/5 (80%) train
# 1/5 (20%) test
from sklearn.model_selection import train_test_split
x = df.iloc[:,0:4] # features # ending index is exclusive
y = df.iloc[:,4] # target (index:5)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# Fit - Train Naive Bayes
print('Train Naive Bayes')
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train, y_train)

# Predict / Test
print('Predict')
y_pred = clf.predict(x_test) # Predicted classes
y_pred_prob = clf.predict_proba(x_test)[:,1] # Probability

# Accurancy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('\nAccuracy Score: ' + str(accuracy))

print('\nConfusion Matrix - Check Accurancy')
# Confusion Matrix - Check Accurancy
confusion_matrix = pd.crosstab(
    y_test,y_pred,
    rownames=['Actual'], colnames=['Prediction'])
print (confusion_matrix)

# Precision, recall and f1-score
print('\nClassification Report - Precision, Recall and F1-Score')
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# ROC Curve
"""
print('\nPlot ROC Curve')
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")
plt.show()
"""


