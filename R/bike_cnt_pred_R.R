##PROBLEM STATEMENT:
#The objective of this project is the Predication of daily bike rental count based on the environmental and seasonal settings.

#Clear Environment
rm(list = ls())

#Set working directory
setwd('C:/Users/Sreejith/Documents/Bike Count Prediction')

#Check Working directory
getwd()

#Load data
df= read.csv('day.csv')

##Exploratory Data Analysis & Data pre-processing:
#Checking the first 5 rows to understand the train data
head(df)

# shape and data types of the data
str(df)

#OBSERVATION:
#We see that 'season','yr','mnth','holiday','weekday','workingday','weathersit' are factors/conditions and therefore we will convert the datatype to factor.
#'dteday' is datetime.

#Datatype correction
df$dteday=strptime(df$dteday,format = "%Y-%m-%d")
cat_var <- c("season","yr","mnth","holiday","weekday","workingday","weathersit")
df[,cat_var] <- lapply(df[,cat_var] , factor)
str(df)

#Summary off the data
summary(df)

#Missig value  Analysis
sum(is.na(df))

#Outlier Analysis
bx_plt<- c("temp","atemp","hum","windspeed")
boxplot(df[,bx_plt])
#boxplot(df$cnt)
##OBSERVATIONS: We see outliers in humidity, windspeed. These might be naturally occuring outliers or errors.
#Lets save the outliers into a vector

#Function to replace outliers with median
outlier <- function(x) {
  x[x < quantile(x,0.25) - 1.5 * IQR(x) | x > quantile(x,0.75) + 1.5 * IQR(x)] <- median(x)
  x
}
#Lets use lapply to replace with median
df[,bx_plt] <- lapply(df[,bx_plt], outlier)
boxplot(df[,bx_plt])

#Missig value  Analysis
sum(is.na(df))

##RELATIONSHIP PLOTS
# temp vs total count 
plot(df$temp, df$cnt, main = "Scatterplot of temp vs cnt", xlab = "Temperature", ylab = "Bike Count")
abline(lm(df$cnt ~ df$temp), col = "blue", lwd = 2)

#atemp vs total count
plot(df$atemp, df$cnt, main = "Scatterplot of atemp vs cnt", xlab = "Feeling Temperature", ylab = "Bike Count")
abline(lm(df$cnt ~ df$atemp), col = "blue", lwd = 2)

#hum vs cnt
plot(df$hum, df$cnt, main = "Scatterplot of hum vs cnt", xlab = "Humidity", ylab = "Bike Count")
abline(lm(df$cnt ~ df$hum), col = "blue", lwd = 2)

#windspeed vs cnt
plot(df$windspeed, df$cnt, main = "Scatterplot of windspeed vs cnt", xlab = "windspeed", ylab = "Bike Count")
abline(lm(df$cnt ~ df$windspeed), col = "blue", lwd = 2)

#Season vs cnt in 2011 (yr=0) and 2012 (yr=1)
library(ggplot2)
ggplot(df, aes(season, cnt, colour = yr)) +
  geom_boxplot()
#OBSERVATION: Here, season 1, 2, 3 and 4 are spring, summer, fall and winter, and yr 0 and 1 are 2011 and 2012 repectively. 
#From the chart its clear that the seasonal effect on count in 2011 and 2012 is same although the count increased in 2012.

## monthly holiday vs total count 
ggplot(df, aes(mnth, cnt, colour = holiday)) +
  geom_boxplot()

# monthly workingday vs total count
ggplot(df, aes(mnth, cnt, colour = workingday)) +
  geom_boxplot()
#OBSERVATIONS: From the above two graghs,taking months 3, 6, and 8 we see that weekends do contribute to the bike count. Even then the working day count overall is more.

# weekday vs total count
ggplot(df, aes(yr, cnt, colour = weekday)) +
  geom_boxplot()
#OBSERVATION: No major effects on the cnt by the days of the week. Weekends have less rides compared to weekdays.

# weather vs total count
ggplot(df, aes(yr, cnt, colour = weathersit)) +
  geom_boxplot()
#OBSERVATION:
#Here, weathersit 1 is clear, Few clouds, Partly cloudy; 
#2 is Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist;
#3 is Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds and
#4 is Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog. 
#Despite the yearly increase in the count, the trend is same here too. Clear weather has more rentals and harsh weather has less rentals.

#FEATURE ENGINEERING
#In this section we will reduce the features on the basis of relevance to the model building and scale them by standardization or normalization.
#backup
backup_df=df
#df=backup_df

#Feature Selection
#Instant and dteday are not useful for the prediction of the bike count as former is the serial number of the data and the later is the date in ascending order and also we already have the useful data from the date like year, month, etc.
df<-subset(df,select=-c(instant,dteday))
#casual and registered count can be dropped as we have the total count as the targeet variable
df<-subset(df,select=-c(casual,registered))

#Correlation Matrix heatmap
#numeric values:
num_var<- subset(df, select=c(temp,atemp,hum,windspeed,cnt))

#Compute the correlation matrix
cormat <- round(cor(num_var),2)
cormat

#install.packages("ggcorrplot")
library(ggcorrplot)
# Add correlation coefficients
# --------------------------------
# argument lab = TRUE
ggcorrplot(cormat, hc.order = TRUE, type = "lower",
           lab = TRUE)
#As expected, the temp and atemp are highly correlated. 

#Feature Selection
#we will select temp and reduce the atemp due to correlation. 
df<-subset(df,select=-c(atemp))

#install.packages("MASS")
#library(MASS)
#table(df$weekday,df$weathersit)
#Chi-sq test
chisq.test(df$season, df$weathersit)
chisq.test(df$season, df$yr)
chisq.test(df$season, df$mnth)
chisq.test(df$season,df$holiday)
chisq.test(df$season,df$weekday)
chisq.test(df$season,df$workingday)

##MODEL BUILDING

#Spliting the features into categorical and numerical 
categorical=c('season','yr','mnth','holiday','weekday','workingday','weathersit')
continuous=c('temp','hum','windspeed')

#Create dummy variables for regression
library(dummies)
df_dummy=dummy.data.frame(df,categorical)
head(df_dummy)

#Defining the predictor variable and the target variable
X=subset(df_dummy,select=-c(cnt))
y=df_dummy['cnt']

## SPLITING DATA TO TEST AND TRAIN
library(caret) #this package has the createDataPartition function

set.seed(123) #randomization`

#creating indices
trainIndex <- createDataPartition(df_dummy$cnt,p=0.75,list=FALSE)

#splitting data into training/testing data using the trainIndex object
X_train <- df_dummy[trainIndex,] #training data (75% of data)

X_test <- df_dummy[-trainIndex,] #testing data (25% of data)


summary(X_train)
summary(X_test)


# Lets apply ML algorithms and select the best fit
##LINEAR REGRESSION

#Model development
LR_model<-lm(cnt~.,data=X_train)
summary(LR_model)
#predict test data by LR model
LR_pred_test=predict(LR_model,X_test)

print(postResample(pred=LR_pred_test,obs = X_test$cnt))
# RMSE        Rsquared        MAE 
#776.0476273  0.8505296     576.3379566


#Calculate MAPE

library(MLmetrics)

LR_mape=MAPE(X_test$cnt,LR_pred_test)
print(LR_mape)
#MAPE= 0.1520104
#Error rate=15.2%
#Accuracy=84.8%


##DECISION TREE REGRESSION
# Install the package
#install.packages("rpart")

# Load the package
library(rpart)
DT_model = rpart(cnt~., data=X_train, method = "anova")
summary(DT_model)
#predict test data by DT model
DT_pred_test=predict(DT_model,X_test)

print(postResample(pred=DT_pred_test,obs = X_test$cnt))
# RMSE        Rsquared        MAE 
#887.7473305  0.8061029     678.4126836


#Calculate MAPE

#library(MLmetrics)

DT_mape=MAPE(X_test$cnt,DT_pred_test)
print(DT_mape)
#MAPE=  0.1864803
#Error rate=18.6%
#Accuracy=81.4%
# Output to be present as PNG file
png(file = "decTree2GFG.png", width = 600,
    height = 600)

# Plot
plot(DT_model, uniform = TRUE,
     main = "Bike Count Decision Tree using Regression")
text(DT_model, use.n = TRUE, cex = .6)

# Saving the file
dev.off()


##RANDOM FOREST REGRESSOR
# Install the required package for function
#install.packages("randomForest")

# Load the library
#library(randomForest)

# Create random forest for regression
RF_model= randomForest(cnt ~ ., data = X_train)
summary(RF_model)

#predict test data by RF model
RF_pred_test=predict(RF_model,X_test)

print(postResample(pred=RF_pred_test,obs = X_test$cnt))
# RMSE        Rsquared        MAE 
#688.042105   0.884561     501.378874 


#Calculate MAPE

#library(MLmetrics)

RF_mape=MAPE(X_test$cnt,RF_pred_test)
print(RF_mape)
#MAPE=  0.1297593
#Error rate=13.0%
#Accuracy=87.0%

# Output to be present as PNG file 
png(file = "randomForestRegression.png")

# Plot the error vs the number of trees graph
plot(RF_model)

# Saving the file
dev.off()


## CLEARLY RANDOM FOREST IS THE BEST MODEL IN PREDICTING THE BIKE COUNT. LETS SEE IF WE CAN IMPROVE THE RESULTS BY PARAMETER TUNING##

##OPTIMIZATION OF THE MODEL
#RANDOM SEARCH
#library(doParallel)
# cores <- 7
# registerDoParallel(cores = cores)
#mtry: Number of random variables collected at each split. In normal equal square number columns.
mtry <- sqrt(ncol(X_train))
#ntree: Number of trees to grow.
ntree <- 3


control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3,
                        search = 'random')

#Random generate 15 mtry values with tuneLength = 15
set.seed(1)
rf_random <- train(cnt ~ .,
                   data = df_dummy,
                   method = 'rf',
                   metric = 'RMSE',
                   tuneLength  = 6, 
                   trControl = control)
print(rf_random)

#predict test data by rf_random model
rf_random_pred_test=predict(rf_random,X_test)

print(postResample(pred=rf_random_pred_test,obs = X_test$cnt))
# RMSE        Rsquared        MAE 
#289.2255373   0.9807255     212.6599990 


#Calculate MAPE

#library(MLmetrics)

rf_random_mape=MAPE(X_test$cnt,rf_random_pred_test)
print(rf_random_mape)
#MAPE=  0.06054491
#Error rate=6%
#Accuracy=94.0%
##WE GOT MAXIMUM ACCURACY AND MIN RMSE FROM THE TUNED MODEL###

###FINAL PREDICTION###
Final_prediction=predict(rf_random,X)

#LETS STORE AS A VARIABLE IN OUR DATASET FOR REFERENCE
df$predicted_count=round(Final_prediction)
head(df)

#SAVE PREDICTION
write.csv(df,file = "Bikecount_predicted_by_R.csv",row.names = F)

#predicted_count vs cnt
plot(df$predicted_count, df$cnt, main = "Scatterplot of predicted_count vs cnt", xlab = "Predicted Bike Count", ylab = "Actual Bike Count")
abline(lm(df$cnt ~ df$predicted_count), col = "blue", lwd = 2)
