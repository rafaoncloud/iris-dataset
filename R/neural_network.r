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

# Binarize the target classes
dfTrain$setosa = dfTrain$species == 'Iris-setosa'
dfTrain$versicolor = dfTrain$species == 'Iris-versicolor'
dfTrain$virginica =  dfTrain$species == 'Iris-virginica'

# Train Neural Network
library(neuralnet)
formula = setosa + versicolor + virginica ~ 
  sepalLength + sepalWidth + petalLength + petalWidth
nn = neuralnet(formula, data= dfTrain, hidden=c(3))
plot(nn)
summary(nn)

# Predict the probability for test data
prediction = compute(nn, dfTest[1:4])
prediction = prediction$net.result

# Consolidate multiple binary output back to categorical output
maxidx = function(arr){
  return(which(arr == max(arr))) 
}
idx = apply(prediction, c(1), maxidx)
resultPredicted = c('setosa,','versicolor','virginica')[idx]
table(resultPredicted, dfTest$species) # Confusion Matrix

