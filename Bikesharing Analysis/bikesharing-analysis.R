## ----setup, include=FALSE----------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(warning = FALSE, message = FALSE)
library(tidyverse)
library(MASS)
library(caret)
library(e1071)
library(GGally)
library(mice)
library(EnvStats)
library(plotly)
library(MetBrewer)
library(pROC)
library(olsrr)
colors = met.brewer('Benedictus')


## ----------------------------------------------------------------------------------------------------------------------------------
data=read.csv(file="SeoulBikeData.csv",
              header=T,sep=",",dec=".", fileEncoding = "ISO-8859-1", stringsAsFactors = TRUE)
glimpse(data)


## ----------------------------------------------------------------------------------------------------------------------------------
sum(is.na(data))


## ----------------------------------------------------------------------------------------------------------------------------------
sum(is.na(data$Rented.Bike.Count))


## ----------------------------------------------------------------------------------------------------------------------------------
data = data[complete.cases(data$Rented.Bike.Count), ]


## ----------------------------------------------------------------------------------------------------------------------------------
data = data[data$Rented.Bike.Count >= 1,]


## ----------------------------------------------------------------------------------------------------------------------------------
for(i in seq(from =4, to = 11)){
  mu <- mean(data[,i])
  sigma <- sd(data[,i])
  data = subset(data, data[,i] > mu - 3*sigma & data[,i] < mu + 3*sigma)
  }
dim(data)


## ----------------------------------------------------------------------------------------------------------------------------------
summary(data$Rented.Bike.Count)


## ----------------------------------------------------------------------------------------------------------------------------------
data$Demand = case_when(
  data$Rented.Bike.Count < 200 ~ 'LowDemand',
  data$Rented.Bike.Count >= 200 & data$Rented.Bike.Count < 1100 ~ 'RegularDemand',
  data$Rented.Bike.Count >= 1100 ~ 'HighDemand'
)


## ----------------------------------------------------------------------------------------------------------------------------------
data = data[,-c(1,14)]


## ----------------------------------------------------------------------------------------------------------------------------------
head(data)


## ----------------------------------------------------------------------------------------------------------------------------------
levels(data$Demand)


## ----------------------------------------------------------------------------------------------------------------------------------
for(i in c(2,11,12,13)){
  data[,i] = as.factor(data[,i])
}
levels(data$Demand)


## ----------------------------------------------------------------------------------------------------------------------------------
spl = createDataPartition(data$Rented.Bike.Count, p = 0.8, list = FALSE)  # 80% for training

train = data[spl,]
test = data[-spl,]


## ----------------------------------------------------------------------------------------------------------------------------------
plot_ly(x = train$Rented.Bike.Count, type = "histogram", color = I(colors[1]))


## ----------------------------------------------------------------------------------------------------------------------------------
ggcorr(train, label = T, label_size = 3, low = colors[1], high = colors[13], midpoint = 0)


## ----------------------------------------------------------------------------------------------------------------------------------
ggplot(train, aes(x=Temperature..C., y=sqrt(Rented.Bike.Count))) + ylab("price") + 
  geom_point(alpha=0.6) + ggtitle("Rented Bike Count vs Temperature")


## ----------------------------------------------------------------------------------------------------------------------------------
par(mfrow= c(3,3))
for(i in c(1,3,4,5,6,7,8,9,10)){
  boxplot(train[,i], las=2, xlab = colnames(train)[i], col = colors[1])
}


## ----------------------------------------------------------------------------------------------------------------------------------
plot_ly(train, labels = ~Demand, values = ~Rented.Bike.Count, type = 'pie', colors = I(colors))


## ----------------------------------------------------------------------------------------------------------------------------------
train = train[,-c(1)]
test = test[,-c(1)]


## ----------------------------------------------------------------------------------------------------------------------------------
lda.model <- lda(Demand ~ ., data=train)

lda.model


## ----------------------------------------------------------------------------------------------------------------------------------
prediction = predict(lda.model, newdata=test)$class
probability = predict(lda.model, newdata=test)$posterior


## ----------------------------------------------------------------------------------------------------------------------------------
confusionMatrix(prediction, test$Demand)$table

## ----------------------------------------------------------------------------------------------------------------------------------
confusionMatrix(prediction, test$Demand)$overall[1]


## ----------------------------------------------------------------------------------------------------------------------------------
data$Seasons = factor(data$Seasons, 
				 levels = c('Autumn', 'Spring', 'Summer', 'Winter'),
				 labels = c(1, 2, 3, 4))
data$Holiday = factor(data$Holiday, 
				 levels = c('Holiday', 'No Holiday'),
				 labels = c(1, 0))

train = data[spl,]
test = data[-spl,]
train = train[,-c(14)]
test = test[,-c(14)]


## ----------------------------------------------------------------------------------------------------------------------------------
corr_count <- sort(cor(train[,c(3,4,5,7,8,9,10)])[1,], decreasing = T)
corr=data.frame(corr_count)
ggplot(corr,aes(x = row.names(corr), y = corr_count)) + 
  geom_bar(stat = "identity", fill = colors[13]) + 
  scale_x_discrete(limits= row.names(corr)) +
  labs(x = "Predictors", y = "Price", title = "Correlations") + 
  theme(plot.title = element_text(hjust = 0, size = rel(1.5)),
        axis.text.x = element_text(angle = 45, hjust = 1))


## ----------------------------------------------------------------------------------------------------------------------------------
linFit <- lm(sqrt(Rented.Bike.Count) ~ Temperature..C., data=train)
summary(linFit)


## ----------------------------------------------------------------------------------------------------------------------------------
par(mfrow=c(2,2))
plot(linFit, pch=23 ,bg=colors[13],cex=2)


## ----------------------------------------------------------------------------------------------------------------------------------
pr.simple = (predict(linFit, newdata=test))^2
cor(test$Rented.Bike.Count, pr.simple)^2


## ----------------------------------------------------------------------------------------------------------------------------------
linFit <- lm(sqrt(Rented.Bike.Count) ~ Hour + Temperature..C. + Humidity... + Wind.speed..m.s. + Visibility..10m. + Dew.point.temperature..C. + Solar.Radiation..MJ.m2. + Rainfall.mm. + Snowfall..cm. + Seasons + Holiday, data=train)
summary(linFit)


## ----------------------------------------------------------------------------------------------------------------------------------
pr.multiple = (predict(linFit, newdata=test))^2
cor(test$Rented.Bike.Count, pr.multiple)^2


## ----------------------------------------------------------------------------------------------------------------------------------
model = sqrt(Rented.Bike.Count) ~ Hour + Temperature..C. + Humidity... + Wind.speed..m.s. + Visibility..10m. + Dew.point.temperature..C. + Solar.Radiation..MJ.m2. + Rainfall.mm. + Snowfall..cm. + Seasons + Holiday

linFit <- lm(model, data=train)

all_predictors = ols_step_all_possible(linFit)

head(all_predictors[order(-all_predictors$rsquare),])


## ----------------------------------------------------------------------------------------------------------------------------------
all_predictors[all_predictors$adjr == max(all_predictors$adjr),]


## ----------------------------------------------------------------------------------------------------------------------------------
plot(ols_step_forward_aic(linFit)) # forward based on AIC


## ----------------------------------------------------------------------------------------------------------------------------------
plot(ols_step_backward_aic(linFit)) # backward AIC


## ----------------------------------------------------------------------------------------------------------------------------------
model = sqrt(Rented.Bike.Count) ~ Hour + Temperature..C. + Humidity... + Wind.speed..m.s. + Visibility..10m. + Dew.point.temperature..C. + Solar.Radiation..MJ.m2. + Rainfall.mm. + Snowfall..cm. + Seasons + Holiday
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5, repeats = 1)


## ----------------------------------------------------------------------------------------------------------------------------------
linFit <- lm(model, data=train)

summary(linFit)


## ----------------------------------------------------------------------------------------------------------------------------------
test_results <- data.frame(Rented.Bike.Count = sqrt(test$Rented.Bike.Count))


## ----------------------------------------------------------------------------------------------------------------------------------
lm_tune <- train(model, data = train, 
                 method = "lm", 
                 preProc=c('scale', 'center'),
                 trControl = ctrl)
lm_tune


## ----------------------------------------------------------------------------------------------------------------------------------
test_results$lm <- predict(lm_tune, test)
postResample(pred = test_results$lm,  obs = test_results$Rented.Bike.Count)


## ----------------------------------------------------------------------------------------------------------------------------------
step_tune <- train(model, data = train, 
                   method = "leapSeq", 
                   preProc=c('scale', 'center'),
                   tuneGrid = expand.grid(nvmax = 4:10),
                   trControl = ctrl)
plot(step_tune)


## ----------------------------------------------------------------------------------------------------------------------------------
test_results$seq <- predict(step_tune, test)
postResample(pred = test_results$seq,  obs = test_results$Rented.Bike.Count)


## ----------------------------------------------------------------------------------------------------------------------------------
lasso_grid <- expand.grid(fraction = seq(.01, 1, length = 100))

lasso_tune <- train(model, data = train,
                    method='lasso',
                    preProc=c('scale','center'),
                    tuneGrid = lasso_grid,
                    trControl=ctrl)
plot(lasso_tune)


## ----------------------------------------------------------------------------------------------------------------------------------
lasso_tune$bestTune


## ----------------------------------------------------------------------------------------------------------------------------------
test_results$lasso <- predict(lasso_tune, test)
postResample(pred = test_results$lasso,  obs = test_results$Rented.Bike.Count)


## ----------------------------------------------------------------------------------------------------------------------------------
elastic_grid = expand.grid(alpha = seq(0, .2, 0.01), lambda = seq(0, .1, 0.01))

glmnet_tune <- train(model, data = train,
                     method='glmnet',
                     preProc=c('scale','center'),
                     tuneGrid = elastic_grid,
                     trControl=ctrl)

plot(glmnet_tune)


## ----------------------------------------------------------------------------------------------------------------------------------
glmnet_tune$bestTune


## ----------------------------------------------------------------------------------------------------------------------------------
test_results$glmnet <- predict(glmnet_tune, test)

postResample(pred = test_results$glmnet,  obs = test_results$Rented.Bike.Count)


## ----------------------------------------------------------------------------------------------------------------------------------
knn_tune <- train(model, 
                  data = train,
                  method = "kknn",   
                  preProc=c('scale','center'),
                  tuneGrid = data.frame(kmax=c(11,13,15,19,21),distance=2,kernel='optimal'),
                  trControl = ctrl)
plot(knn_tune)


## ----------------------------------------------------------------------------------------------------------------------------------
test_results$knn <- predict(knn_tune, test)

postResample(pred = test_results$knn,  obs = test_results$Rented.Bike.Count)


## ----------------------------------------------------------------------------------------------------------------------------------
apply(test_results[-1], 2, function(x) mean(abs(x - test_results$Rented.Bike.Count)))


## ----------------------------------------------------------------------------------------------------------------------------------
yhat = (test_results$knn)^2

hist(yhat, col=colors[1])


## ----------------------------------------------------------------------------------------------------------------------------------
y = (test_results$Rented.Bike.Count)^2
error = y-yhat
hist(error, col=colors[13])


## ----------------------------------------------------------------------------------------------------------------------------------
noise = error[1:100]
lwr = yhat[101:length(yhat)] + quantile(noise,0.05, na.rm=T)
upr = yhat[101:length(yhat)] + quantile(noise,0.95, na.rm=T)
predictions = data.frame(real=y[101:length(y)], fit=yhat[101:length(yhat)], lwr=lwr, upr=upr)
predictions = predictions %>% mutate(out=factor(if_else(real<lwr | real>upr,1,0)))

ggplot(predictions, aes(x=fit, y=real))+
  geom_point(aes(color=out)) + theme(legend.position="none") +
  geom_ribbon(data=predictions,aes(ymin=lwr,ymax=upr),alpha=0.3) +
  labs(title = "Prediction intervals", x = "prediction",y="real price")

