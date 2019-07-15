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

# Train the Support Vector Machine
library(e1071)
formula = species ~ 
  sepalLength + sepalWidth + petalLength + petalWidth
tune = tune.svm(formula, data=dfTrain, gamma=10^(-6:-1), cost=10^(1:4))
summary(tune)

model = svm(formula, dfTrain, method='C-classification', 
            kernel='radial', probability=T, gamma=0.001, cost= 10000)

# Test the Support Vector Machine
prediction = predict(model, dfTest, probability=T)

# Confusion Matrix
table(prediction, dfTest$species)

