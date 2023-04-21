library(glmnet)
candidate_lengths <- c(7, 9, 11)

Rcpp::sourceCpp('src/rocket.cpp')

generate_kernels <- function(input_length, num_kernels) {
  lengths <- sample(candidate_lengths, num_kernels, replace=TRUE)
  weights <- numeric(sum(lengths))
  biases <- runif(num_kernels, min=-1, max=1)
  dilations <- numeric(num_kernels)
  paddings <- numeric(num_kernels)
  index <- 1
  for (i in 1:num_kernels) {
    length_ <- lengths[i]
    weights_ <- rnorm(length_)
    next_index <- index + length_

    weights[index:(next_index-1)] <- weights_ - mean(weights_)

    dilation <- 2^(runif(1, min=0, max=max(0, log2((input_length - 1) / (length_ - 1)))))
    dilation <- floor(dilation)
    dilations[i] <- dilation
    if (sample(c(TRUE, FALSE), 1)){
      padding <- floor(((length_ - 1) * dilation)/2)
    }else{
      padding <- 0
    }
    paddings[i] <- padding
    index <- next_index
  }
  c(weights, lengths, biases, dilations, paddings)
}

l2_norm <- function(x) {
    sqrt(sum(x*x))
}


dataset_names <- c("Trace", "WormsTwoClass", "Worms", "Coffee",
                   "Computers", "Crop", "FaceAll",
                   "Fish", "Earthquakes", "Yoga")

dataset_names <- c("Trace", "WormsTwoClass", "Worms", "Coffee",
                   "Computers", "FaceAll",
                   "Fish", "Earthquakes", "Yoga")
lambdas <- c(1000.0,  215.44346900318823,  46.41588833612773,  10.0,  2.154434690031882,  0.46415888336127775,  0.1,
             0.021544346900318832,  0.004641588833612777,  0.001)
num_kernels <- 10000
n_trails <- 1
for (dataset_name in dataset_names){

  test_filename <- paste("./datasets/", dataset_name, "_TEST.txt", sep="")
  train_filename <- paste("./datasets/", dataset_name, "_TRAIN.TXT", sep="")

  test <- read.csv(test_filename, header=FALSE, sep=",")
  train <- read.csv(train_filename, header=FALSE, sep=",")
  train <- train[sample(nrow(train)),]

  y_train <- train$V1
  y_test <- test$V1

  train <- train[,2:length(train)]
  std = sqrt(apply(train, 1, var))
  train <- sweep(train, MARGIN=1, STATS = rowMeans(train))
  train <- sweep(train, MARGIN=1, STATS = std, FUN = "/")


  test <- test[,2:length(test)]
  std = sqrt(apply(test, 1, var))
  test <- sweep(test, MARGIN=1, STATS = rowMeans(test))
  test <- sweep(test, MARGIN=1, STATS = std, FUN = "/")

  print(c(dataset_name, nrow(train)))
  test_accuracies <- numeric(0)
  train_accuracies <- numeric(0)

  for (i in 1:n_trails){
    print(i)

    kernels_data <- generate_kernels(length(test), num_kernels)


    test_transformed <- apply_kernels(data.matrix(test), kernels_data, num_kernels)
    test_transformed <- do.call(rbind.data.frame, test_transformed)

    train_transformed <- apply_kernels(data.matrix(train), kernels_data, num_kernels)
    train_transformed <- do.call(rbind.data.frame, train_transformed)


    means <- colMeans(train_transformed)
    l2_norms <- apply(train_transformed, 2, l2_norm)
    train_transformed <- sweep(train_transformed, MARGIN=2, STATS = means)
    train_transformed <- sweep(train_transformed, MARGIN=2, STATS = l2_norms, FUN = "/")

    X_train <- as.matrix(train_transformed[,1:2])

    test_transformed <- sweep(test_transformed, MARGIN=2, STATS = means)
    test_transformed <- sweep(test_transformed, MARGIN=2, STATS = l2_norms, FUN = "/")

    X_test <- as.matrix(test_transformed[,1:2])

    ridge_reg <- cv.glmnet(X_train, y_train, alpha = 0, type.measure = "class", nfolds = 5,
                           family = "multinomial", lambda = lambdas, type.multinomial = "grouped")

    train_predicted <- predict(ridge_reg, newx=X_train, s=ridge_reg$lambda.min, type = "class")
    test_predicted <- predict(ridge_reg, newx=X_test, s=ridge_reg$lambda.min, type = "class")

    train_accuracies <- c(train_accuracies, sum(train_predicted==y_train)/nrow(train))
    test_accuracies <- c(test_accuracies, sum(test_predicted==y_test)/nrow(test))
  }
  print(c(mean(train_accuracies), mean(test_accuracies)))
  print(train_accuracies)
  print(test_accuracies)

}

