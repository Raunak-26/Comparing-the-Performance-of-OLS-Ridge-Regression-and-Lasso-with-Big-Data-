set.seed(101)

#Setting up the simulation

load("C:/Users/Raunak Mehrotra/Desktop/Raunak_CompStats_Project/data.RData")
#install.packages("glmnet")
library(glmnet)
library(ggplot2)
library(reshape2)
library(knitr)

rep = 100     #Number of repetitions
train_MSE=matrix(NA,rep,3) #MSE on train set
test_MSE = matrix(NA,rep,3) #MSE on test set, that is, prediction error
colnames(test_MSE)=c("OLS","Ridge","Lasso")
colnames(train_MSE)=c("OLS","Ridge","Lasso")


sample = sample(1:nrow(d), size = round(0.75*nrow(d)), replace=FALSE) #Train:Test data split
train  = d[sample,]
test   = d[-sample,]
y.train=train$y #Response variable in train data
x.train=train[,-1] #Predictors in train data
x.train=as.matrix(x.train)
y.test=test$y #Response variable in test data
x.test=test[,-1] #Predictors in test data
x.test=as.matrix(x.test)
data.train=data.frame(y.train,x.train) #Training data
data.test = data.frame(y.test,x.test)  #Testing data

grid = 10^seq(10,-2,length=100) #Length of values for lambda used in Ridge regression and LASSO

#Correlation Matrix Heatmap

cor1 = round(cor(subset(d,select=X100:X135)),2) #Correlation matrix
upper_tri = function(cor1){
  cor1[lower.tri(cor1)]=0
  return(cor1)
} #Lower triangle matrix
upper_cor1 = upper_tri(cor1) 
meltcor = melt(upper_cor1)
heatmap=ggplot(data=meltcor,aes(x=Var2,y=Var1,fill=value))+geom_tile(color="white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  ggtitle("Correlation Matrix Heatmap")+
  theme_minimal()+ # minimal theme
  theme(
    plot.title = element_text(color="black", size=14, face="bold.italic",vjust=-1,hjust=0.5),
    axis.text.x = element_text(angle = 90, vjust = 0.5, 
                               size = 10, hjust = 1))+
  coord_fixed()

heatmap + 
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5)) 

#Least Squares Regression
set.seed(101)
for (i in 1:rep){
  
  lm.train = lm(y.train ~ x.train-1,data=data.train) #Linear regression on train data
  lm.train$coefficients[is.na(lm.train$coefficients)]=0 #Setting 'NA' values due to multicollinearity to zero to calculate estimates
  lm.train.predict = x.train%*%lm.train$coefficient #Response estimates on train data
  lm.test.predict = x.test%*%lm.train$coefficient   #Response estimates on test data
  train_MSE[i,1] = mean((lm.train.predict-y.train)^2) #MSE on train data
  test_MSE[i,1] = mean((lm.test.predict-y.test)^2) #Prediction error
}

par(mfrow=c(1,2)) #Response-Fitted values/Predicted values plot
plot(y.train,lm.train.predict,col='green',main = "Training",xlab ="Response",ylab = "Fitted Values",lwd=2) 
abline(lm(lm.train.predict~y.train),col='red')
plot(y.test,lm.test.predict,col='green',main="Testing",xlab ="Response",ylab = "Prediction",lwd=2) 
abline(lm(lm.test.predict~y.test),col='red')
title(sub="Linear Regression", line=-1,outer = TRUE)

#Ridge regression

set.seed(101)
for (i in 1:rep){
  
  lambda.ridge = cv.glmnet(x.train,y.train,lambda = grid,alpha=0,intercept=FALSE) #10-fold Cross-validation for calculation optimal value of lambda 
  opt.lambda.ridge = lambda.ridge$lambda.min #Optimal value of lambda
  ridge.train = glmnet(x.train,y.train,lambda = grid,alpha=0,intercept=FALSE) #Ridge regression on train data using a length of values of lambda
  ridge.train.pred = predict(ridge.train,s=opt.lambda.ridge,newx=x.train) #Response estimates on train data using optimal lambda value
  ridge.test.pred = predict(ridge.train,s=opt.lambda.ridge,newx=x.test) #Response estimates on test data using optimal lambda value
  train_MSE[i,2] = mean((ridge.train.pred - y.train)^2) #MSE on train data
  test_MSE[i,2] = mean((ridge.test.pred - y.test)^2) #Prediction error
}

par(mfrow=c(1,2)) #Response-Fitted values/Predicted values plot
plot(y.train,ridge.train.pred,col='green',main = "Training",xlab ="Response",ylab = "Fitted Values",lwd=2) 
abline(lm(ridge.train.pred~y.train),col='red')
plot(y.test,ridge.test.pred,col='green',main="Testing",xlab ="Response",ylab = "Prediction",lwd=2) 
abline(lm(ridge.test.pred~y.test),col='red')
title(sub="Ridge Regression", line=-1,outer = TRUE)

par(mfrow=c(1,1))
plot(lambda.ridge,xlim=c(-5,10)) #Cross validation for optimal lambda plot
title(sub="Cross-validation for Optimal Lambda Plot", line=-1,outer = TRUE,)
print(paste("The optimal value of lambda for ridge regression is",round(opt.lambda.ridge,3)))

#LASSO

set.seed(101)
for (i in 1:rep){
  
  lambda.lasso = cv.glmnet(x.train,y.train,lambda = grid,alpha=1,intercept=FALSE) #10-fold Cross-validation for calculation optimal value of lambda
  opt.lambda.lasso = lambda.lasso$lambda.min  #Optimal value of lambda
  lasso.train = glmnet(x.train,y.train,lambda = grid,alpha=1,intercept=FALSE) #LASSO on train data using a length of values of lambda
  lasso.train.pred = predict(lasso.train,s=opt.lambda.lasso,newx=x.train) #Response estimates on train data using optimal lambda value
  lasso.test.pred = predict(lasso.train,s=opt.lambda.lasso,newx=x.test) #Response estimates on test data using optimal lambda value
  train_MSE[i,3] = mean((lasso.train.pred - y.train)^2) #MSE on train data
  test_MSE[i,3] = mean((lasso.test.pred - y.test)^2) #Prediction error
}

par(mfrow=c(1,2))  #Response-Fitted values/Predicted values plot
plot(y.train,lasso.train.pred,col='green',main = "Training",xlab ="Response",ylab = "Fitted Values",lwd=2) 
abline(lm(lasso.train.pred~y.train),col='red')
plot(y.test,lasso.test.pred,col='green',main="Testing",xlab ="Response",ylab = "Prediction",lwd=2) 
abline(lm(lasso.test.pred~y.test),col='red')
title(sub="Lasso Regression", line=-1,outer = TRUE)

par(mfrow=c(1,1))
plot(lambda.lasso,xlim=c(-5,2)) #Cross validation for optimal lambda plot
title(sub="Cross-validation for Optimal Lambda Plot", line=-1,outer = TRUE)
print(paste("The optimal value of lambda for lasso is",round(opt.lambda.lasso,3)))
outcome = coef(lasso.train, s=opt.lambda.lasso)
kable(outcome[outcome[,1]!=0,],caption="Non-zero Predictors") #Non-zero predictors in train data using optimal lambda value


