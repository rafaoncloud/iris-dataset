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

# Train 100 trees, random selected attributes
library(randomForest)
formula = species ~ sepalLength + sepalWidth + petalLength + petalWidth
model = randomForest(formula, data=dfTrain, nTree=100)
summary(model)
plot(model)

# Predict using the decision tree (Test)
prediction = predict(model, newdata=dfTest[1:4], type='class')

# Confusion Matrix 
library(caret)
caret::confusionMatrix(prediction, dfTest$species)

# Importance of each attribute
importance(model)

