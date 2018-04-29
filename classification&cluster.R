library(mice)
library(randomForest)
library(xgboost)
library(cluster)
file_train <- ('./data/train.csv')
file_test  <- ('./data/test.csv')
data_train <- read.csv(file_train, header = T, sep = ',')
data_test  <- read.csv(file_test , header = T, sep = ',')
data <- rbind(data_train,data_test)
data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$SibSp <- as.factor(data$SibSp)
data$Parch <- as.factor(data$Parch)
# 缺失值处理
data <- data[-c(1,4,9,11)]
data$Embarked[c(62,830)] <- 'C'
data$Fare[1044] <- median(data[data$Pclass == '3' & data$Embarked == 'S', ]$Fare, na.rm = TRUE)
set.seed(666)
mice_mod <- mice(data[c(2:8)], method='cart')
mice_output <- complete(mice_mod)
data$Age <- mice_output$Age

train <- data[1:891,]
test  <- data[892:1309,]
formula <- Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked
#随机森林分类
rf_model <- randomForest(formula,train)
plot(rf_model, ylim=c(0,0.4))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
varImpPlot(rf_model)
pred <- predict(rf_model,test)
write.csv(pred,file = 'pred.csv')
#kaggle score 0.76555

target_rf <- train$Survived
pred_train_rf <- predict(rf_model,train)
table(pred_train_rf,target_rf)

#xgboost 分类
train_data <- data.matrix(train[c(-1)])
test_data <- data.matrix(test[c(-1)])
target <- data.matrix(train[1])
xgb_model <- xgboost(data = train_data,label = target,
                     eta=0.1,max_depth = 5,nrounds = 100)
# rmse 100 -> 0.237
pred2 <- predict(xgb_model,test_data)
pred_xgb <- ifelse(pred2>1.5,1,0)
write.csv(pred_xgb,file = 'pred_xgb.csv')
#kaggle score 0.77511

#结果可视化
pred_train <- predict(xgb_model,train_data)
pred_train <- ifelse(pred_train>1.5,2,1)
table(pred_train,target)
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix) 
#legend('topright',colnames(importance),col =0:6 ,fill = c('Sex','Fare','Age','Pclass','SibSp','Embarked','Parch'))


cluster_data <- data[-c(1)]
train <- cluster_data[1:891,]
test  <- cluster_data[892:1309,]
target <- data$Survived[c(1:891)]

# k-means
km <- kmeans(data.matrix(train),centers = 3,nstart = 10)
fitted(km)
#结果可视化
table(target,km$cluster)
plot(train[c("Age","Fare")], col = km$cluster, pch = as.integer(target))    
points(km$centers[,c("Age","Fare")], col = 1:3, pch = 8, cex=2); 

#Fuzzy Analysis Clustering
fc <- fanny(data.matrix(train),3,metric = 'SqEuclidean')
clusplot(fc)
table(target,fc$clustering)

