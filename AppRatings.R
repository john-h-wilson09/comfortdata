if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(recosystem)

# https://www.kaggle.com/gauthamp10/google-playstore-apps

dl <- tempfile()
download.file("https://github.com/john-h-wilson09/comfortdata/raw/master/androidapps.zip", dl)

dat <- read.csv(unzip(dl,"googleplaystore.csv"))
rm(dl)

str(dat) #13 variables, last four variables appear useless
sum(is.na(dat)) #checking for NAs, many missing values
dat1 <- na.omit(dat) #remove NAs

# Number of unique apps
length(unique(dat1$App))

# Number of unique genres
length(unique(dat1$Genres))

dat1 %>% group_by(Price) %>% summarize(n=n()) %>% 
  arrange(desc(n)) %>% head(10) #most apps are free or under $3

dat1 %>% count(Installs) %>% ggplot(aes(n)) + geom_histogram(color="black") + 
  labs(title = "Installs", x="") #not all have the same amount of installs
# Seeing the top installed apps
dat1 %>% arrange(desc(Installs)) %>% tail(10) %>% select(App,Category,Rating,Reviews,Installs,Price)

dat1 %>% group_by(Reviews) %>% summarize(n=n()) %>% 
  arrange(desc(n)) %>% head(10) #most apps have less than 5 reviews
dat1 %>% group_by(Rating) %>% summarize(n=n()) %>% arrange(desc(n)) %>%
  head(5) #most apps are rated 4.3-4.5

dat1 %>% count(Category) %>% ggplot(aes(n)) + geom_histogram(color="black") + 
  labs(title = "Category", x="") #not all categories are equally represented
dat1 %>% group_by(Category) %>% summarize(n=n()) %>% arrange(desc(n)) %>%
  head(5) #top 5 categories for playstore apps

qplot(dat1$Rating,dat1$Reviews) #no relationship noticed between number of reviews and rating
dat1 <- dat1 %>% filter(Rating<6) #filter out outlier

# Dividing the data to have a practice set and and validation set
set.seed(15,sample.kind = "Rounding")
valid_index <- createDataPartition(dat1$Rating,times = 1,list = FALSE,p=0.15)
practice <- dat1[-valid_index,]
validation <- dat1[valid_index,]

# Making sure valid set info is in practice set also
validation <- validation %>% semi_join(practice, by = "Genres") %>% semi_join(practice, by = "Category") %>%
  semi_join(practice, by = "Content.Rating")

# Dividing the practice set to have data to train and test with
test_index <- createDataPartition(practice$Rating,times = 1,list = FALSE,p=0.2)
train_set <- practice[-test_index,]
test_set <- practice[test_index,]

# Making sure the test set info is also in the train set
test_set <- test_set %>% semi_join(train_set, by = "Genres") %>% semi_join(train_set, by = "Category") %>%
  semi_join(train_set, by = "Content.Rating")

mu <- mean(train_set$Rating)
avg_pred <- rep(mu,length(test_set$Rating))
avg_rmse <- RMSE(avg_pred,test_set$Rating)
RMSE_result0 <- data_frame(Model = "Naive-Avg", RMSE = avg_rmse) #create table for results

cat_avg <- train_set %>% group_by(Category) %>% summarize(b_c=mean(Rating-mu))
cat_pred <- test_set %>% left_join(cat_avg,by='Category') %>%
  mutate(pred=mu+b_c) %>% pull(pred)
cat_rmse <- RMSE(cat_pred,test_set$Rating)
RMSE_result1 <- bind_rows(RMSE_result0, data_frame(Model="Category Effect Model", RMSE = cat_rmse ))

genre_avg <- train_set %>% left_join(cat_avg, by='Category') %>%
  group_by(Genres) %>% summarize(b_g=mean(Rating-mu-b_c))
genre_pred <- test_set %>% left_join(cat_avg,by='Category') %>%
  left_join(genre_avg,by='Genres') %>% mutate(pred=mu+b_c+b_g) %>%
  pull(pred)
genre_rmse <- RMSE(genre_pred,test_set$Rating)
RMSE_result1 <- bind_rows(RMSE_result1, data_frame(Model="Category+Genre Effect Model", RMSE = genre_rmse ))

content_avg <- train_set %>% left_join(cat_avg,by='Category') %>%
  left_join(genre_avg,by='Genres') %>% group_by(Content.Rating) %>%
  summarize(b_r = mean(Rating-mu-b_c-b_g))
content_pred <- test_set %>% left_join(cat_avg,by='Category') %>%
  left_join(genre_avg,by='Genres') %>%
  left_join(content_avg,by='Content.Rating') %>% mutate(pred=mu+b_c+b_g+b_r) %>%
  pull(pred)
content_rmse <- RMSE(content_pred,test_set$Rating)
RMSE_result1 <- bind_rows(RMSE_result1, data_frame(Model="Cat+Genre+Content Model", RMSE = content_rmse ))

lambdas <- seq(2,10,0.25)
RMSE_reg <- sapply(lambdas, function(l){
  cat_avg <- train_set %>% group_by(Category) %>% summarize(b_c=sum(Rating-mu)/(n()+l))
  genre_avg <- train_set %>% left_join(cat_avg, by='Category') %>%
    group_by(Genres) %>% summarize(b_g=sum(Rating-mu-b_c)/(n()+l))
  
  preds <- test_set %>% left_join(cat_avg,by='Category') %>%
    left_join(genre_avg,by='Genres') %>% mutate(pred=mu+b_c+b_g) %>%
    pull(pred)
  
  RMSE(preds,test_set$Rating)
})
qplot(lambdas,RMSE_reg)
lambda <- lambdas[which.min(RMSE_reg)] #best lamda 
RMSEreg_eff <- RMSE_reg[[which.min(RMSE_reg)]]
RMSE_result2 <- bind_rows(RMSE_result1, data_frame(Model="Regularized Eff Model", RMSE = RMSEreg_eff ))

# Matrix Factorization Model
# Calculating residuals by removing movie, user and genres effect
residual_set <- train_set %>% left_join(cat_avg,by='Category') %>%
  left_join(genre_avg,by='Genres') %>% mutate(resid=Rating-mu-b_c-b_g) 

# Setting data and tuning parameters - used R documentation for parameters
reco <- Reco()
train_reco <- data_memory(user_index=residual_set$Category, item_index=residual_set$Genres, 
                          rating=residual_set$resid, index1=TRUE)
test_reco <- data_memory(user_index=test_set$Category, item_index=test_set$Genres, index1=TRUE)

# This step takes a couple of minutes
opts <- reco$tune(train_reco, opts = list(dim = c(10,20,30,40,50), 
                                          lrate = c(0.1, 0.2),
                                          costp_l1=0, costq_l1=0,
                                          nthread = 1, niter = 10))

# Train the model of reco
reco$train(train_reco, opts = c(opts$min, nthread=1, niter=20))

# Predict results for the test set
reco_pred <- reco$predict(test_reco, out_memory())
preds <- cbind(test_set, reco_pred) %>% 
  left_join(cat_avg,by='Category') %>% left_join(genre_avg,by='Genres') %>% 
  mutate(pred = mu + b_c + b_g + reco_pred) %>% pull(pred)
rmseMF <- RMSE(preds, test_set$Rating)
RMSE_result3 <- bind_rows(RMSE_result2, data_frame(Model="Matrix Factorization", RMSE = rmseMF )) %>%
  arrange(desc(RMSE)) #arranging RMSE from largest to smallest to see best model
RMSE_result3 #best model was matrix factorization

# Apply best model validation data
valid <- validation %>% semi_join(train_set, by = "Genres") %>% semi_join(train_set, by = "Category") 
valid_reco <- data_memory(user_index=valid$Category, item_index=valid$Genres, index1=TRUE)
reco_val <- reco$predict(valid_reco, out_memory())
valid_pred <-  cbind(valid, reco_val) %>% 
  left_join(cat_avg,by='Category') %>% left_join(genre_avg,by='Genres') %>% 
  mutate(vpred = mu + b_c + b_g + reco_val) %>% pull(vpred)
valid_rmse <- RMSE(valid_pred,valid$Rating)
valid_rmse
