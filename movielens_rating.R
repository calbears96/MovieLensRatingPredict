#the following is the R script that is found (mostly) in the Rmd document

#first, load the necessary libraries
library(tidyverse)
library(caret)
library(lubridate)
library(irlba)
library(recosystem)
library(recommenderlab)
library(ggplot2)
library(kableExtra)

#the next section of code is from the edx site, no alterations were made
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
        semi_join(edx, by = "movieId") %>%
        semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#EXPLORATORY DATA ANALYSIS

#simply look at the head (first 6 observations of the edx dataset)
head(edx)

#summary of the edx data set
summary(edx)

#create a histogram of the ratings in the edx data
#use dplyr (tidyverse) to pipe the dataset and then plot histogram
edx %>% select(rating) %>% ggplot(aes(x=rating)) +
        geom_histogram(binwidth=.2, color='black', fill='blue') +
        scale_x_continuous(breaks=seq(0,5, by=.5)) +
        labs(x='Rating', y='Number of ratings')

#create table of genres with genre and count as the columns
genres = edx %>% separate_rows(genres, sep = "\\|") %>%
        group_by(genres) %>%
        summarize(count = n()) %>%
        arrange(desc(count))

#create new dataframe called top movies, summarized by the count (observations)
#then take that dataframe and pipe it to ggplot and create a bar graph of the
#top movies
top_movies = edx %>% group_by(title) %>%
        summarize(count = n()) %>%
        top_n(10, count) %>%
        arrange(desc(count))

top_movies %>% ggplot(aes(x = reorder(title, count), y=count)) +
        geom_bar(stat='identity', fill='red') + coord_flip(y=c(0,35000)) +
        labs(x = "", y='Number of ratings') 

#create a histogram of the number of ratings by movieId
edx %>% count(movieId) %>%
        ggplot(aes(n)) +
        geom_histogram(bins=30, color='black', fill='red') +
        scale_x_log10()+
        labs(x = 'movieId', y = 'Number of ratings')

#create a histogram of the number of ratings by userId
edx %>% count(userId) %>%
        ggplot(aes(n)) +
        geom_histogram(bins=30, color='black', fill='blue') +
        scale_x_log10() +
        labs(x = 'userId',
             y = 'Number of ratings')

#create a graph of the mean rating by unit of time, in this case the week
edx %>% mutate(date = round_date(as_datetime(timestamp), unit = 'week')) %>%
        group_by(date) %>%
        summarize(rating = mean(rating)) %>%
        ggplot(aes(date, rating)) +
        geom_point() +
        geom_smooth() +
        labs(x = 'Date', y = 'Mean rating')

#DATA TRANSFORMATION/MATRIX FACTORIZATION SECTION

#create a copy of the edx dataset, since we are going to be messing around with it
#keeping only three variables: userId, movieId, and rating
edx_copy = subset(edx, select = userId:rating)

#thought this might be useful...but probably not
edx_copy$userId = as.factor(edx_copy$userId)
edx_copy$movieId = as.factor(edx_copy$movieId)

#sparsematrix, must convert userId and movieId to numeric
edx_copy$userId = as.numeric(edx_copy$userId)
edx_copy$movieId = as.numeric(edx_copy$movieId)

#create the sparse matrix
ratings = sparseMatrix(i = edx_copy$userId,
                       j = edx_copy$movieId,
                       x = edx_copy$rating,
                       dims = c(length(unique(edx_copy$userId)),
                                length(unique(edx_copy$movieId))),
                       dimnames = list(paste('u', 1:length(unique(edx_copy$userId)), sep=''),
                                       paste('m', 1:length(unique(edx_copy$movieId)), sep='')))

#remove the edx_copy
rm(edx_copy)

#now convert rating matrix into the r package 'recommenderlab' sparse matrix (real rating)
ratings_matrix = new('realRatingMatrix', data=ratings)

#begin the dimension reduction through the use of a partial SVD.
#Y (matrix of N x P) can be decomposed into UDV^T where
# U = ortogonal matrix of dimension N x m
# D = diagonal matrix containing singular values of original matrix Y
# V = orthogonal matrix of dimension m x P
set.seed(23)
y = irlba(ratings, tol=1e-4, verbose=TRUE, nv=100, maxit=1000)

#sum squares of singular values
all_sq = sum(y$d^2)

#variability by singular values
first_six = sum(y$d[1:6]^2)
#print(first_six / all_sq)

percent_vec = NULL
for (i in 1:length(y$d)) {
        percent_vec[i] = sum(y$d[1:i]^2) / all_sq
}

#plot showing k for dimensionality reduction of the matrix and sum squared singular
#values
plot(percent_vec, pch=20, col='red', cex=1.5, xlab='Singular value', ylab='% SS of 
     singular values')
lines(x = c(0,100), y=c(.9,.9))

#define soe of the variables--not used later though for dimensionality reduction
k = length(percent_vec[percent_vec <=.9])

U_k = y$u[, 1:k]

D_k = Diagonal(x=y$d[1:k])

V_k = t(y$v)[1:k,]

#find minimum number of movies per user
min_movies = quantile(rowCounts(ratings_matrix), .9)

#find minimum number of users per movie
min_users = quantile(colCounts(ratings_matrix), .9)

#select users with those criteria for # movies, # users, create new matrix
ratings_movies = ratings_matrix[rowCounts(ratings_matrix) > min_movies,
                                colCounts(ratings_matrix) > min_users]

##MODELING AND RESULTS

#starting with the linear model 
#define the average rating of the data (mu)
mu = mean(edx$rating)

#create new variable (b_i) in the movie_avgs dataframe. Defined as the average
#rating of a movie, adjusting for the average rating for all movies
movie_avgs = edx %>%
        group_by(movieId) %>%
        summarize(b_i = mean(rating - mu))

#predicted ratings using above and combining with the validation data set
predicted_ratings_bi = mu + validation %>%
        left_join(movie_avgs, by ='movieId') %>%
        .$b_i

#now add in a user effect for the second linear model
#movie and user effect
user_avgs = edx %>%
        left_join(movie_avgs, by='movieId') %>%
        group_by(userId) %>%
        summarize(b_u = mean(rating - mu - b_i))

predicted_ratings_bu = validation %>%
        left_join(movie_avgs, by='movieId') %>%
        left_join(user_avgs, by='userId') %>%
        mutate(pred = mu + b_i + b_u) %>%
        .$pred

#calculate the RMSE for both linear models
rmse_model = RMSE(validation$rating, predicted_ratings_bi)

rmse_model2 = RMSE(validation$rating, predicted_ratings_bu)

#regularization model to penalize "bias" from small numbers of ratings
lambdas = seq(0, 10, .25) #tuning parameter

rmses = sapply(lambdas, function(l) {
        
        mu_reg = mean(edx$rating)
        
        bi_reg = edx %>%
                group_by(movieId) %>%
                summarise(bi_reg = sum(rating - mu_reg)/(n()+l))
        
        bu_reg = edx %>%
                left_join(bi_reg, by='movieId') %>%
                group_by(userId) %>%
                summarise(bu_reg = sum(rating - bi_reg - mu_reg)/(n()+l))
        
        predicted_ratings_biu = validation %>%
                left_join(bi_reg, by='movieId') %>%
                left_join(bu_reg, by='userId') %>%
                mutate(pred = mu_reg + bi_reg + bu_reg) %>%
                .$pred
        
        return(RMSE(validation$rating, predicted_ratings_biu))
})

#plot the lambdas and RMSEs, visually inspect for optimal lambda
qplot(lambdas, rmses)

#find optimal lambda
lambda = lambdas[which.min(rmses)]

#find the mimimum RMSE
rmse_model3 = min(rmses)

#recommender models
#try popular method
model_pop = Recommender(ratings_movies, method='POPULAR',
                        param = list(normalize='center'))

predict_pop = predict(model_pop, ratings_movies, type='ratings')

set.seed(23)
e = evaluationScheme(ratings_movies, method='split', train=.7, given=-5)
model_pop = Recommender(getData(e, 'train'), 'POPULAR')

prediction_pop = predict(model_pop, getData(e, 'known'), type='ratings')

#calculate RMSE for popular method
rmse_popular = calcPredictionAccuracy(prediction_pop, getData(e, 'unknown'))[1]

#try User Based Collaborative Filtering (UBCF)
set.seed(23)
model = Recommender(getData(e, 'train'), method='UBCF',
                    param = list(normalize='center', method='Cosine', nn=50))

prediction = predict(model, getData(e, 'known'), type='ratings')

#rmse for UBCF
rmse_ubcf = calcPredictionAccuracy(prediction, getData(e, 'unknown'))[1]

#MATRIX FACTORIZATOIN
invisible(gc())

#create a copy of the edx data set with only the userId, movieId, and rating
edx_copy = edx %>% select(c('userId', 'movieId', 'rating'))

#renaming
names(edx_copy) = c('user', 'item', 'rating')

#make a copy of the validation data set, keeping only userId, movieId, and rating
validation_copy = validation %>%
        select(c('userId', 'movieId', 'rating'))

names(validation_copy) = c('user', 'item', 'rating')

#convert to matrices
edx_copy = as.matrix(edx_copy)
validation_copy = as.matrix(validation_copy)

write.table(edx_copy, file='~/Documents/trainset.txt', sep=' ', row.names=FALSE, col.names = FALSE)
write.table(validation_copy, file='~/Documents/validationset.txt', sep= ' ', row.names = FALSE, col.names = FALSE)

set.seed(23)
#make sure the data sets are in the recosystem format
train_set <- data_file( "~/Documents/trainset.txt" , package = "recosystem")
validation_set = data_file('~/Documents/validationset.txt', package='recosystem')

r = Reco()

#tune the training set
opts = r$tune(train_set, opts=list(dim=c(10,20, 30), lrate=c(.1,.2),
                                   costp_l1 = 0, costql1=0,
                                   nthread = 1, niter =10))

#now we train the recommender model
r$train(train_set, opts=c(opts$min, nthread = 4, niter=100, verbose=FALSE))

#create the prediction file
pred_file = tempfile()

#predict the ratings
r$predict(validation_set, out_file(pred_file))

#print(scan(pred_file, n=10))

scores_real = read.table('~/Documents/validationset.txt', header=FALSE, sep = " ")$V3
scores_pred = scan(pred_file)

rm(edx_copy, validation_copy)

#calculate RMSE for matrix factorization
rmse_mf_opt = RMSE(scores_real, scores_pred)
