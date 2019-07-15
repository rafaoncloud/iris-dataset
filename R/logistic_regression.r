# Rafael Henriques
# 15-Jul-2019

# Read Dataset
df = read.table("../dataset/iris.data", header=TRUE, sep=",")

summary(df)
head(df)

# Add a new column - is the iris specie setosa (binary) ?
df$isSetosa = df$species == 'Iris-setosa'
# Convert from binary FALSE and TRUE to 0 and 1
df$isSetosa = as.numeric(df$isSetosa)

# Prepare the training and testing data
# 4/5 (80%) train
# 1/5 (20%) test
testIdx = which(1:length(df[,1]) %% 5 == 0)

dfTrain = df[-testIdx,]
dfTest = df[testIdx,]

summary(dfTrain)
summary(dfTest)

# Create formula (target ~ features)
formula = isSetosa ~ sepalLength + sepalWidth + petalLength + petalWidth

# Logistic Regression - Fit
logisticModel = glm(formula, data=dfTrain, family='binomial')
summary(logisticModel)

# Predict the probability for test data
prob = predict(logisticModel, newdata=dfTest, type='response')
round(prob, 3)

# Confusion Matrix
library(caret)
caret::confusionMatrix(factor(as.numeric(prob > 0.5)), factor(dfTest$isSetosa))

# ROC (Receiver Operator Characteristic Curve)
library(ROCR)
ROCRpred = prediction(prob, dfTest$isSetosa)
ROCRperf = performance(ROCRpred, "tpr", "fpr") # Performance function
plot(ROCRperf, colorize=TRUE) # Plot ROC curve

