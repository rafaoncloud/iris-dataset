# Rafael Henriques
# 15-Jul-2019

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt

# Read Dataset
df = pd.read_csv("../dataset/iris.data")

print(df.shape)
print(df.info())
print(df.head(4));

# Remove Unacessary Data
df = df[['petalLength','petalWidth']].copy()
print(df.head(2))

# Prepare the training and testing data
# 4/5 (80%) train
# 1/5 (20%) test
from sklearn.model_selection import train_test_split
x = df.iloc[:,0:1] # feature # ending index is exclusive
y = df.iloc[:,1:2] # target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# Predict / Test
y_predict = model.predict(x_test)

# Plot
plt.plot(x_test,y_predict)
plt.scatter(x_test,y_predict)
plt.scatter(x_test,y_test, color="orange")
plt.show()