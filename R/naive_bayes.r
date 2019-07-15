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

# Train the Naive Bayes
# Can't handle both categorical and numeric variables, 
# altough output must be categorical
library(e1071)
formula = species ~ sepalLength + sepalWidth + petalLength + petalWidth
model = naiveBayes(formula, data=dfTrain, usekernel = TRUE)
summary(model)

# Test the Naive Bayes
prediction = predict(model, dfTest[1:4])

# Confusion Matrix
table(dfTest$species, prediction)

