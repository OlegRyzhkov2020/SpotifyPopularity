#load relevant libraries
library(ranger)
library(caret)
library(dplyr)
library(xgboost)
library(h2o)

h2o.init(nthreads=-1, max_mem_size="8G")
h2o.no_progress()


#load data
target.df <- h2o.importFile('https://raw.githubusercontent.com/OlegRyzhkov2020/SpotifyPopularity/main/data/target_numeric_popularity.csv') 
splits <- h2o.splitFrame(target.df, c(0.6,0.2), seed=1)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%

df.train <- as.data.frame(train)
df.valid <- as.data.frame(valid)
df.test  <- as.data.frame(test)


#========================================================================================
#============================= Random Forest ============================================
#========================================================================================

hyper_grid <- expand.grid(
  mtry       = seq(9, 15, by = 2),
  node_size  = c(1, 5, 10),
  OOB_RMSE   = 0
)

for(i in 1:nrow(hyper_grid)) {
  
  model <- ranger(
    formula         = popularity ~ ., 
    data            = df.train, 
    num.trees       = 1000,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    seed            = 345,
  )
  
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
  print(i)
}

(oo = hyper_grid %>% 
    dplyr::arrange(OOB_RMSE) %>%
    head(10))

rf.fit.final <- ranger(
  formula         = popularity ~ ., 
  data            = df.train,  
  num.trees       = 1000,
  mtry            = oo[1,]$mtry,
  min.node.size   = oo[1,]$node_size,
)

yhat.rf = predict(rf.fit.final, data = df.test)$predictions
rmse.rf = sqrt(mean((df.test$popularity-yhat.rf)^2))
#7.45070
r2.rf   = 1 - sum((df.test$popularity-yhat.rf)^2)/sum((df.test$popularity-mean(df.test$popularity))^2)
#0.93173


#========================================================================================
#============================= Boosted ==================================================
#========================================================================================

#no need to turn into model matrix since data was already one-hot encoded by h2o
X.train = as.matrix(df.train[,2:25])
X.test = as.matrix(df.test[,2:25])
Y.train = df.train$popularity

hyper_grid_xgb <- expand.grid(
  shrinkage = c(0.001, .01, .1, 1),         ## controls the learning rate
  interaction.depth = c(1, 2, 4), ## tree depth
  bag.fraction = c(.5, .65, .8, .9),  ##  percent of training data to sample for each tree
  optimal_trees = 0,              # a place to dump results
  min_RMSE = 0                    # a place to dump results
)

for(i in 1:nrow(hyper_grid_xgb)) {
  (i)
  # create parameter list
  params <- list(
    eta = hyper_grid_xgb$shrinkage[i],
    max_depth = hyper_grid_xgb$interaction.depth[i],
    subsample = hyper_grid_xgb$bag.fraction[i]
  )
  
  # reproducibility
  set.seed(41654)
  
  # train model
  xgb.tune <- xgb.cv(
    params               = params,
    data                 = X.train,
    label                = Y.train,
    nrounds              = 5000,
    nfold                = 5,
    objective            = "reg:squarederror",
    verbose              = 0,
    verbosity            = 0,
    nthread              = 5,
    early_stopping_rounds = 10     # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid_xgb$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid_xgb$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)  
}

(oo = hyper_grid_xgb %>%
    dplyr::arrange(min_RMSE) %>%
    head(10))

# parameter list
params <- list(
  eta = oo[1,]$shrinkage,
  max_depth = oo[1,]$interaction.depth,
  subsample = oo[1,]$bag.fraction
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X.train,
  label = Y.train,
  nrounds = oo[1,]$optimal_trees,
  objective = "reg:squarederror",
  verbose = 0,
  verbosity = 0
)

yhat.xgb = predict(xgb.fit.final, newdata=X.test)
rmse.xgb = sqrt(mean((df.test$popularity-yhat.xgb)^2))
#7.87906
r2.xgb   = 1 - sum((df.test$popularity-yhat.xgb)^2)/sum((df.test$popularity-mean(df.test$popularity))^2)
#0.92365

