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

# Fit - Train 
print('Generate KMeans Cluster')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x_train[['petalLength','petalWidth']])

# Predict / Test
print('Predict')
y_pred = kmeans.predict(x_test[['petalLength','petalWidth']]) # Predicted classes

#Plot the clusters obtained using k-means
y_kmeans = kmeans.predict(x_train[['petalLength','petalWidth']])
plt.scatter(x_train['petalLength'], x_train['petalWidth'], c=y_kmeans)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()

