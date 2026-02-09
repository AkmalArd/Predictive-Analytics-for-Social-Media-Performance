
library(tidyverse)
library(readr)
library(fastDummies)
library(glmnet)  


train_target <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Train_Target.csv")
train_predictor <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Training_Data_Predictors.csv")
test_predictor <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Test_Data_Predictors.csv")


train_target <- as_tibble(train_target)
train_predictor <- as_tibble(train_predictor)
test_predictor <- as_tibble(test_predictor)


train_predictor$Type <- as.factor(train_predictor$Type)
test_predictor$Type <- as.factor(test_predictor$Type)


train_predictor <- dummy_cols(train_predictor, select_columns = "Type", remove_selected_columns = TRUE)
test_predictor <- dummy_cols(test_predictor, select_columns = "Type", remove_selected_columns = TRUE)

# test data has all columns present in training data
missing_cols <- setdiff(names(train_predictor), names(test_predictor))
for (col in missing_cols) {
  test_predictor[[col]] <- 0
}
test_predictor <- test_predictor[, names(train_predictor)]  

if ("ID" %in% names(train_target)) {
  train_target <- train_target[, !(names(train_target) %in% "ID"), drop = FALSE]
}

train_data <- cbind(train_target, train_predictor)


train_data <- na.omit(train_data)

#  Ridge Regression
X_train <- as.matrix(train_data[, -1])  
y_train <- train_data$Total_Interactions


set.seed(123)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)  


best_lambda <- cv_ridge$lambda.min


ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)


X_test <- as.matrix(test_predictor)


test_predictor$.pred <- predict(ridge_model, newx = X_test, s = best_lambda)


test_predictor$.pred <- pmax(0, test_predictor$.pred)


predictions <- test_predictor[, c("ID", ".pred")]
write.csv(predictions, "Ridge_Regression_Predictions.csv", row.names = FALSE)
