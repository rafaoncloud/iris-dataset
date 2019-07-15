# Rafael Henriques
# 15-Jul-2019

# Read Dataset
df = read.table("../dataset/iris.data", header=TRUE, sep=",")

summary(df)
head(df)

# Prepare the training and testing data
# 4/5 (80%) train
# 1/5 (20%) test
testIdx = which(1:length(df[,1]) %% 5 == 0)

dfTrain = df[-testIdx,]
dfTest = df[testIdx,]

summary(dfTrain)
summary(dfTest)

# Prediction using the k-nearest-neightbor - Unsupervised Learning
library(class)
formula = species ~ sepalLength + sepalWidth + petalLength + petalWidth
prediction = knn(dfTrain[1:4],dfTest[1:4],dfTrain$species,k=5)

# Confusion Matrix
table(dfTest$species, prediction)

