library(data.table)
library(lubridate)
library(dplyr)
library(ggplot2)
library(tidyr)
library(MASS)
library(caret)

setwd('final_project')

train = fread('train100k.csv', stringsAsFactors = F)
glimpse(train)

train1 <- fread('train_all.csv', stringsAsFactors = F)

train2 <- train1[1:1000000,]

test1 <- train1[1000001:1500000,]

### Logistic Regression ###

mod1 <- glm(as.numeric(is_attributed) ~ channelIp10s + channelApp10s + channelDevice10s + channelOs10s + 
      osIp10s + osApp10s + osDevice10s + appAttrib + deviceAttrib + osAttrib + channelAttrib +
      ip10s + app10s + device10s + os10s + channel10s, data = train_bal, family = 'binomial')

mod2 <- glm(is_attributed ~ channelIp3s + channelApp3s + channelDevice3s + channelOs3s + 
              osIp3s + osApp3s + osDevice3s + appAttrib + deviceAttrib + osAttrib + channelAttrib +
              ip30s + app30s + device30s + os30s + channel30s, data = train2, family = 'binomial')

mod1.prob <- predict(mod1, newdata = test1, type = 'response')
mod1.predict <- ifelse(mod1.prob > .01, 1,0)
table(test1$is_attributed,mod1.predict)

library(ROCR)

rocr_pred <- prediction(mod1.prob, test1$is_attributed)
rocr_perf <- performance(rocr_pred, "tpr", "fpr")
performance(rocr_pred,measure='auc')

roc_df <- data.frame(fp_rate = rocr_perf@x.values[[1]], tp_rate = rocr_perf@y.values[[1]])
roc_df %>% 
  ggplot(aes(x=fp_rate,y=tp_rate))+
  geom_line(color='green3', size=1)+
  stat_function(fun = function(x) x)+
  theme_bw()

### LDA ###

mod3 <- lda(is_attributed ~ channelIp10s + channelApp10s + channelDevice10s + channelOs10s + 
              osIp10s + osApp10s + osDevice10s + appAttrib + deviceAttrib + osAttrib + channelAttrib +
              ip10s + app10s + device10s + os10s + channel10s, data = train2)

mod3.pred <- predict(mod3, newdata = test1)
mod3.class <- mod3.pred$class
table(test1$is_attributed,mod3.class)

rocr_pred <- prediction(mod3.pred$posterior[,2], test1$is_attributed)
rocr_perf <- performance(rocr_pred, "tpr", "fpr")
performance(rocr_pred,measure='auc')

roc_df <- data.frame(fp_rate = rocr_perf@x.values[[1]], tp_rate = rocr_perf@y.values[[1]])
roc_df %>% 
  ggplot(aes(x=fp_rate,y=tp_rate))+
  geom_line(color='green3', size=1)+
  stat_function(fun = function(x) x)+
  theme_bw()

### Random Forest ###

library(randomForest)

mod4 <- randomForest(as.factor(is_attributed) ~ channelIp10s + channelApp10s + channelDevice10s + channelOs10s + 
                       osIp10s + osApp10s + osDevice10s + appAttrib + deviceAttrib + osAttrib + channelAttrib +
                       ip10s + app10s + device10s + os10s + channel10s, data=train2, ntree=50)

mutate(data.frame(importance(mod4)),attr=rownames(data.frame(importance(mod4)))) %>% 
  ggplot(aes(x=reorder(attr,MeanDecreaseGini),y=MeanDecreaseGini))+
  geom_col(fill='dodgerblue2')+
  coord_flip()

mod4.predict <- predict(mod4, newdata = test1)

table(test1$is_attributed, mod4.predict)

rocr_pred <- prediction(mod4.predict, test1$is_attributed)
rocr_perf <- performance(rocr_pred, "tpr", "fpr")
performance(rocr_pred,measure='auc')

roc_df <- data.frame(fp_rate = rocr_perf@x.values[[1]], tp_rate = rocr_perf@y.values[[1]])
roc_df %>% 
  ggplot(aes(x=fp_rate,y=tp_rate))+
  geom_line(color='green3', size=1)+
  stat_function(fun = function(x) x)+
  theme_bw()


#rfCtrl1 = trainControl(method='repeatedcv', number = 5, repeats = 5)
set.seed(555)
rf = train(Survived ~ .,
           data = titanic.train,
           method = 'rf',
           trControl = rfCtrl1)
           
train_bal <- fread('train_balanced.csv', stringsAsFactors = F)

############################
#hybrid sampling using SMOTE

library(dplyr)
library(tidyr)
library(caret)
library(DMwR)
library(ROCR)

train_all <- fread('train_all.csv', stringsAsFactors = F)

# change is_attributed to factor, remove original vars to prepare for upsampling
train_all <- train_all %>% 
  mutate(is_attributed = factor(is_attributed),
         channel = factor(channel),
         os = factor(os),
         device = factor(device),
         app = factor(app),
         ip = factor(ip),
         click_time = factor(click_time))

#remove rows pripor to 1min
train1 <- train_all[58316:1000000,]
test1 <- train_all[1000001:1500000,]

table(train1$is_attributed)

set.seed(100)
smote_train <- SMOTE(is_attributed ~ ., data = train1, perc.over = 1000, perc.under = 1000)

table(smote_train$is_attributed)

#check equivilency
train1 %>% 
  dplyr::select(is_attributed, contains('channel'), -channel) %>% 
  gather(var, val, -is_attributed) %>% 
  group_by(var, is_attributed) %>% 
  summarise(mean_val = mean(val)) %>% 
  spread(is_attributed, mean_val)

#Test with Logistic Regression using 10s predictors
mod1 <- glm(is_attributed ~ channelIp10s + channelApp10s + channelDevice10s + channelOs10s + 
              osIp10s + osApp10s + osDevice10s + appAttrib + deviceAttrib + osAttrib + channelAttrib +
              ip10s + app10s + device10s + os10s + channel10s, data = smote_train, family = 'binomial')

mod1.prob <- predict(mod1, newdata = test1, type = 'response')
mod1.predict <- ifelse(mod1.prob > .5, 1,0)
table(test1$is_attributed,mod1.predict)

rocr_pred <- prediction(mod1.prob, test1$is_attributed)
rocr_perf <- performance(rocr_pred, "tpr", "fpr")
performance(rocr_pred,measure='auc')

roc_df <- data.frame(fp_rate = rocr_perf@x.values[[1]], tp_rate = rocr_perf@y.values[[1]])
roc_df %>% 
  ggplot(aes(x=fp_rate,y=tp_rate))+
  geom_line(color='green3', size=1)+
  stat_function(fun = function(x) x)+
  theme_bw()

#write.csv(smote_train, 'train_smote.csv', row.names = F)
library(readr)

smote_train <- read_csv('train_smote_Apr15.csv')

### reg subsets

library(leaps)

exclude <- c("record","channel","os","device","app","ip","click_time")
smote_train$is_attributed <- factor(smote_train$is_attributed)

smote_sub <- smote_train[,-which(names(smote_train) %in% exclude)]

system.time(best.subset <- regsubsets(is_attributed~., smote_sub, really.big = T, nvmax = 12))

best.subset.summary = summary(best.subset)

best.subset.by.adjr2 = which.max(best.subset.summary$adjr2)
best.subset.by.adjr2

best.subset.by.cp = which.min(best.subset.summary$cp)
best.subset.by.cp

best.subset.by.bic = which.min(best.subset.summary$bic)
best.subset.by.bic

filter(data.frame(feature = names(best.subset.summary$which[11,]),
                  result = best.subset.summary$which[11,]), result == T)


library(ggplot2)
library(dplyr)
library(tidyr)
library(extrafont)

labeler <- function(x){
  if(x[1] < -1){
  return(paste0(x/1000,'k'))
  }
  else if(x[1] == 0){
    return(c('0','10k','20k',NA))
  }
  else{
    return(x)
  }
}

data.frame(Rsq = best.subset.summary$adjr2, 
           BIC = best.subset.summary$bic,
           Cp = best.subset.summary$cp,
           rn = 0:11) %>% 
  gather(metric,value,-rn) %>% 
  ggplot(aes(x=rn, y = value))+
  geom_line()+
  facet_grid(metric~., scales = 'free_y')+
  scale_x_continuous(breaks = seq(0,11,1))+
  scale_y_continuous(labels = function(x) labeler(x))+
  theme_bw()+
  theme(text= element_text(family = 'Arial', size=16),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = .5))+
  labs(x='Number of Predictors', y = '', title = 'Best Subset Selection')
