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

data.frame(pred = test_pred, act = cs_test$Sales) %>% 
  mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))

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

#bagging with random forest
library(randomForest)

bag.carseats <- randomForest(Sales~., data=Carseats)
yhat.bag <- predict(bag.carseats,newdata=cs_test)

# data.frame(pred = yhat.bag, act = cs_test$Sales) %>% 
#   mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))

mean((yhat.bag -cs_test$Sales)^2)
importance(bag.carseats)

#rf with 

#caret tree implementation



#linear regression
lr <- lm(Sales~., data=cs_train)
lr_pred <- predict(lr,cs_test)

data.frame(pred = lr_pred, act = cs_test$Sales) %>% 
  mutate(sqerr = (pred-act)^2) %>% summarise(mean(sqerr))
