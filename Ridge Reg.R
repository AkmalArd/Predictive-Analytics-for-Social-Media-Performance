# Load necessary libraries
library(tidyverse)
library(readr)
library(fastDummies)
library(glmnet)  # Ridge & Lasso Regression

# Load datasets
train_target <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Train_Target.csv")
train_predictor <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Training_Data_Predictors.csv")
test_predictor <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Test_Data_Predictors.csv")

# Convert to tibble for compatibility
train_target <- as_tibble(train_target)
train_predictor <- as_tibble(train_predictor)
test_predictor <- as_tibble(test_predictor)

# Convert categorical variables into factors
train_predictor$Type <- as.factor(train_predictor$Type)
test_predictor$Type <- as.factor(test_predictor$Type)

# One-hot encoding for "Type" column
train_predictor <- dummy_cols(train_predictor, select_columns = "Type", remove_selected_columns = TRUE)
test_predictor <- dummy_cols(test_predictor, select_columns = "Type", remove_selected_columns = TRUE)

# Ensure test data has all columns present in training data
missing_cols <- setdiff(names(train_predictor), names(test_predictor))
for (col in missing_cols) {
  test_predictor[[col]] <- 0
}
test_predictor <- test_predictor[, names(train_predictor)]  # Align column order

# ✅ Remove ID column from train_target (Fix for select(-ID) error)
if ("ID" %in% names(train_target)) {
  train_target <- train_target[, !(names(train_target) %in% "ID"), drop = FALSE]
}

# Combine target and predictor variables
train_data <- cbind(train_target, train_predictor)

# Handle missing values
train_data <- na.omit(train_data)

# Convert to matrix for Ridge Regression
X_train <- as.matrix(train_data[, -1])  # Remove response variable
y_train <- train_data$Total_Interactions

# Train Ridge Regression model with cross-validation
set.seed(123)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)  # alpha = 0 → Ridge Regression

# Get best lambda (optimal penalty term)
best_lambda <- cv_ridge$lambda.min

# Fit final Ridge model
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)

# Convert test set to matrix
X_test <- as.matrix(test_predictor)

# Make predictions on test data
test_predictor$.pred <- predict(ridge_model, newx = X_test, s = best_lambda)

# Ensure non-negative predictions
test_predictor$.pred <- pmax(0, test_predictor$.pred)

# Save predictions
predictions <- test_predictor[, c("ID", ".pred")]
write.csv(predictions, "Ridge_Regression_Predictions.csv", row.names = FALSE)
