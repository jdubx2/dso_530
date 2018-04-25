library(ISLR)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(tree)
data("Carseats")

set.seed(2)

train_idx <- sample(nrow(Carseats),200)

cs_train <- Carseats[train_idx,]
cs_test <- Carseats[-train_idx,]

#base tree implementation

tree.carseats <- tree(Sales~.,data=cs_train)
summary(tree.carseats)

plot(tree.carseats )
text(tree.carseats ,pretty =0)

test_pred <- predict(tree.carseats, newdata = cs_test)

mean((test_pred -cs_test$Sales)^2)

# data.frame(pred = test_pred, act = cs_test$Sales) %>% 
#   mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))

#pruning with cross validation
cvtree.carseats <- cv.tree(tree.carseats)
data.frame(size = cvtree.carseats$size, dev = cvtree.carseats$dev) %>% 
  ggplot(aes(x = size,y = dev))+
  geom_line()+scale_x_continuous(breaks = seq(1,17,1))

pruning <- function(tree_model,sizes){
  
  for(size in sizes){if(size !=1){
    prune.carseats = prune.tree(tree.carseats, best = size)
    
    test_pred <- predict(prune.carseats, newdata = cs_test)
    
    mse <- data.frame(pred = test_pred, act = cs_test$Sales) %>% 
      mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))
    
    print(paste0(size,' Node Test MSE: ',round(mse,3)))
  }}
}

pruning(tree.carseats,cvtree.carseats$size)

#bagging with randomForest
library(randomForest)

bag.carseats <- randomForest(Sales~., data=cs_train,mtry=10)
yhat.bag <- predict(bag.carseats,newdata=cs_test)

# data.frame(pred = yhat.bag, act = cs_test$Sales) %>% 
#   mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))

mean((yhat.bag -cs_test$Sales)^2)
importance(bag.carseats)

#rf with randomForest

cv_rf <- function(train,test){
  
  result_list <- list()
  
  for(m in seq(1:(ncol(train)-1))){
    trained <- randomForest(Sales~.,data=train, mtry=m)
    yhat <- predict(trained, newdata=test)
    
    result_list[[m]] <- data.frame(mtry=m, mse = mean((yhat-test$Sales)^2))
  }
  return(bind_rows(result_list))
}

rf.carseats <- cv_rf(cs_train,cs_test)

rf.carseats %>% 
  ggplot(aes(x = mtry, y = mse))+
  geom_line()


#caret tree implementation



#linear regression
lr <- lm(Sales~., data=cs_train)
lr_pred <- predict(lr,cs_test)

data.frame(pred = lr_pred, act = cs_test$Sales) %>% 
  mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))


lr <- lm(Sales~.-Population-Education-Urban-US, data=cs_train)

library(FactoMineR)
pca = PCA(Carseats[,c(2,3,4,5,6,8,9)], scale.unit=TRUE, ncp=5, graph=T)


library(leaps)
library(glmnet)

x <- model.matrix(Sales~.,data=cs_train)[,c(2,3,4,6,7,8,9)]
y <- cs_train$Sales

grid=10^seq(10,-2, length =100)

ridge.mod <- glmnet(x,y,alpha=0, lambda=grid, standardize=T)

ridge_pred=predict(ridge.mod ,s=.03, newx= model.matrix(Sales~.,data=cs_test)[,c(2,3,4,6,7,8,9)])

data.frame(pred = ridge_pred, lr = lr_pred, act = cs_test$Sales) %>% 
  mutate(sqerr = (X1-act)^2) %>% summarise(mean(sqerr))
