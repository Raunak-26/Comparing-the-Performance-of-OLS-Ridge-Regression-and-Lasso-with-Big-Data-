set.seed(101)

#Setting up the study

load("C:/Users/Raunak Mehrotra/Desktop/Raunak_CompStats_Project/Real.2.rda")
library(glmnet)
library(knitr)

x=as.matrix(Real.2)[,-1:-2] #Predictors in entire dataset
y = Real.2$y #Response variable
data = data.frame(y,x) #Entire dataset

sample = sample(1:nrow(data), size = round(0.60*nrow(data)), replace=FALSE) #Train:Test data split
train  = data[sample,]
test   = data[-sample,]
x.train = model.matrix(y~.,data=train)[,-1] #Predictors in train data
x.test = model.matrix(y~.,data=test)[,-1] #Predictors in test data
y.train = train$y #Response variable in train data
y.test = test$y #Response variabe in test data
data.train=data.frame(y.train,x.train) #Training data
data.test = data.frame(y.test,x.test) #Test data

grid = 10^seq(10,-2,length=100) #Length of values for lambda used in Ridge regression and LASSO

#Ridge regression

set.seed(101)

model.ridge = glmnet(x,y,alpha=0) #Ridge regression on entire dataset using a length of values of lambda chosen by the function
plot(model.ridge,xvar="lambda") #Coefficients shrinkage path
title(sub="Coefficients shrinkage path", line=-1,outer = TRUE)

cv.ridge = cv.glmnet(x.train,y.train,lambda = grid,alpha=0) #10-fold Cross-validation for ridge regression model and calculation optimal value of lambda on train data
opt.lambda_ridge = cv.ridge$lambda.min #Optimal value of lambda
round(opt.lambda_ridge,3)

ridge.train.pred = predict(cv.ridge,s=opt.lambda_ridge,newx=x.train) #Response estimates on train data using optimal lambda value
Ridge_train_MSE = mean((ridge.train.pred - y.train)^2) #MSE on train data
round(Ridge_train_MSE,3)
ridge.test.pred = predict(cv.ridge,s=opt.lambda_ridge,newx=x.test) #Response estimates on test data using optimal lambda value
Ridge_test_MSE = mean((ridge.test.pred - y.test)^2) #Prediction error
round(Ridge_test_MSE,3)

#LASSO

set.seed(101)

model.lasso = glmnet(x,y,alpha=1) #LASSO on entire dataset using a length of values of lambda chosen by the function
cv.lasso1 = cv.glmnet(x,y,lambda = grid,alpha=1) #10-fold Cross-validation for calculation of optimal value of lambda on entire data
opt.lambda_lasso1 = cv.lasso1$lambda.min #Optimal value of lambda
plot(model.lasso,xvar="lambda") #Coefficients shrinkage path
abline(v =log(opt.lambda_lasso1),col="Black",lty=2) #Selected model based on optimal lambda value
title(sub="Coefficients shrinkage path", line=-1,outer = TRUE)
outcome1 = coef(model.lasso, s=opt.lambda_lasso1)
kable(outcome1[outcome1[,1]!=0,],caption="Non-zero Predictors") #Non-zero predictors in entire data using optimal lambda value

cv.lasso = cv.glmnet(x.train,y.train,lambda = grid,alpha=1) #10-fold Cross-validation for LASSO model and calculation optimal value of lambda on train data
opt.lambda_lasso = cv.lasso$lambda.min #Optimal value of lambda
round(opt.lambda.lasso,3)
plot(cv.lasso,xlim=c(-5,2))  #Cross validation for optimal lambda plot
title(sub="Cross validation for optimal lambda plot", line=-1,outer = TRUE)

lasso.train.pred = predict(cv.lasso,s=opt.lambda_lasso,newx=x.train) #Response estimates on train data using optimal lambda value
Lasso_train_MSE = mean((lasso.train.pred - y.train)^2) #MSE on train data
round(Lasso_train_MSE,3)
lasso.test.pred = predict(cv.lasso,s=opt.lambda_lasso,newx=x.test) #Response estimates on test data using optimal lambda value
Lasso_test_MSE = mean((lasso.test.pred - y.test)^2) #Prediction error
round(Lasso_test_MSE,3)
