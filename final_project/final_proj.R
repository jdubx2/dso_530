library(data.table)
library(lubridate)
library(dplyr)
library(ggplot2)
library(MASS)

setwd('final_project')

train = fread('train100k.csv', stringsAsFactors = F)
glimpse(train)

train1 <- fread('train_all.csv', stringsAsFactors = F)

train2 <- train1[1:1000000,]

test1 <- train1[1000001:1500000,]

### Logistic Regression ###

mod1 <- glm(is_attributed ~ channelIp10s + channelApp10s + channelDevice10s + channelOs10s + 
      osIp10s + osApp10s + osDevice10s + appAttrib + deviceAttrib + osAttrib + channelAttrib +
      ip10s + app10s + device10s + os10s + channel10s, data = train2, family = 'binomial')

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
