##Load all library
library(ipred);
library(plyr);
library(gbm);
library(survival);
library(MASS);
library(klaR);
library(caret);
library(pROC);
library(rpart);


#Read the data from training file
pml <- read.csv("~/Documents/pml-training.csv", header = TRUE, na.strings = c("", "NA", "#DIV/0!"))

#Cleanup of data and irrelevaent columns
pml1 <- pml
indx <- sapply(pml1, is.logical)
pml1[indx] <- lapply(pml1[indx], function(x) as.numeric(as.numeric(x)))
pml2 <- pml1
pml3<-pml2[ , ! apply( pml2 , 2 , function(x) all(is.na(x)) ) ]
pml4<-pml3[ , ! apply( pml3 , 2 , function(x) sum(is.na(x))/sum(1)>0.4 ) ]
pml5 <- pml4[,8:60]

#Split the data between training and test/validation set
inTrain <- createDataPartition(y=pml5$classe, p=0.75, list=FALSE)
training <- pml5[inTrain,]
testing <- pml5[-inTrain,]


#Set reproduceable random number
set.seed(13343)

#set training control
trCtrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

#Train model using Random Forest Algorithm
modelrf <- train(classe ~ ., data=training, method="rf", trControl=trCtrl, prox=FALSE)
confusionMatrix(testing$classe, predict(modelrf, testing))
plot(varImp(modelrf))


#Train using Bagged CART Algorithm

modelbag <- train(classe ~ ., data=training, method="treebag", trControl=trCtrl)
confusionMatrix(testing$classe, predict(modelbag, testing))
plot(varImp(modelbag))


#Traing using Gradient Boosting Model
modelgbm <- train(classe ~ ., data=training, method="gbm", trControl=trCtrl, 
                  verbose=FALSE)
confusionMatrix(testing$classe, predict(modelgbm, testing))
plot(varImp(modelgbm))

#Training using Quadratic Discriminant Analysis 
modelqda <- train(classe ~ ., data=training, method="qda", trControl=trCtrl)
confusionMatrix(testing$classe, predict(modelqda, testing))
plot(varImp(modelqda))


compare <- list("Random Forest"=modelrf$results[2, 2:5], 
                "Bagged CART"=modelbag$results[, 2:5], 
                "Gradient Boosting"=modelgbm$results[9, 4:7], 
                "Quadratic Discriminant Analysis"=modelqda$results[, 2:5] 
                )
compare <- data.frame(compare)
compare


#Load Submission Test Data
pmlsub <- read.csv("~/Documents/pml-testing.csv", header = TRUE, na.strings = c("", "NA", "#DIV/0!"))

indx <- sapply(pmlsub, is.logical)
pmlsub[indx] <- lapply(pmlsub[indx], function(x) as.numeric(as.numeric(x)))
pmlsub<-pmlsub[ , ! apply( pmlsub , 2 , function(x) all(is.na(x)) ) ]
pmlsub<-pmlsub[ , ! apply( pmlsub , 2 , function(x) sum(is.na(x))/sum(1)>0.4 ) ]

#Remove the unnecessary columns from the validation data
pmlsub <- pmlsub[,8:60]


results <- t(data.frame(randomForest = predict(modelrf, pmlsub), 
                        bag = predict(modelbag, pmlsub),
                        boost = predict(modelgbm, pmlsub),
                        qda = predict(modelqda, pmlsub)))
results


#writing output file
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("~/Documents/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

#pml_write_files(results[1,])