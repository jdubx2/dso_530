library(dplyr)
library(tidyr)
library(caret)
library(DMwR)
library(ROCR)

train_all <- fread('train_all.csv', stringsAsFactors = F)

# change categorical vars to factors
train_all <- train_all %>% 
  mutate(is_attributed = factor(is_attributed),
         channel = factor(channel),
         os = factor(os),
         device = factor(device),
         app = factor(app),
         ip = factor(ip),
         click_time = factor(click_time))

#remove observations priors to 1min
train1 <- train_all[58316:1000000,]
test1 <- train_all[1000001:1500000,]

table(train1$is_attributed)

set.seed(100)
smote_train <- SMOTE(is_attributed ~ ., data = train1, perc.over = 1000, perc.under = 1000)

table(smote_train$is_attributed)

#check similiartiy of channel columns
# smote_train %>% 
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

#input/output
#write.csv(smote_train, 'train_smote_Apr15.csv', row.names = F)
#smote_train <- read.csv('train_smote_Apr15.csv', stringsAsFactors = F)

### reg subsets ###

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