# Rafael Henriques
# 15-Jul-2019

# Read Dataset
df = read.table("../dataset/iris.data", header=TRUE, sep=",")

summary(df)

head(df)

# Correlation Matrix
library(GGally)
ggpairs(df)

# Remove unnecessary columns (features)
df$sepalLength = NULL
df$sepalWidth = NULL



# Prepare the training and testing data
# 4/5 (80%) train
# 1/5 (20%) test
testIdx = which(1:length(df[,1]) %% 5 == 0)

dfTrain = df[-testIdx,]
dfTest = df[testIdx,]

summary(dfTrain)
summary(dfTest)

# Linear Regression
model = lm(formula=petalWidth ~ petalLength, data=dfTrain)

# Use the model to predict the output of test data
prediction = predict(model,newdata=dfTest)
summary(prediction)

cor(prediction, dfTest$petalWidth)
summary(model)

# Petal width dependent on sepal width
plot(x=dfTrain$petalLength, y=dfTrain$petalWidth, main="petalWidth ~ petalLength")
# Regression linear function
abline(model)


