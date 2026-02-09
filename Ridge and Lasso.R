# 📦 Libraries
library(tidyverse)
library(readr)
library(fastDummies)
library(glmnet)
library(Metrics)      
library(DataExplorer) 
library(corrplot)     

# 📁 Load datasets
train_target     <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Train_Target.csv")
train_predictor  <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Training_Data_Predictors.csv")
test_predictor   <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Test_Data_Predictors.csv")


train_predictor <- train_predictor %>%
  mutate(Type_Photo  = ifelse(Type == "Photo", 1, 0),
         Type_Link   = ifelse(Type == "Link", 1, 0),
         Type_Status = ifelse(Type == "Status", 1, 0)) %>%
  dplyr::select(-Type)

test_predictor <- test_predictor %>%
  mutate(Type_Photo  = ifelse(Type == "Photo", 1, 0),
         Type_Link   = ifelse(Type == "Link", 1, 0),
         Type_Status = ifelse(Type == "Status", 1, 0)) %>%
  dplyr::select(-Type)

train_predictor <- train_predictor %>%
  mutate(Category_1 = ifelse(Category == 1, 1, 0),
         Category_2 = ifelse(Category == 2, 1, 0)) %>%
  dplyr::select(-Category)

test_predictor <- test_predictor %>%
  mutate(Category_1 = ifelse(Category == 1, 1, 0),
         Category_2 = ifelse(Category == 2, 1, 0)) %>%
  dplyr::select(-Category)
numeric_cols <- train_predictor %>% select_if(is.numeric) %>% names()
dummy_cols   <- c("Type_Photo","Type_Link","Type_Status","Category_1","Category_2","Paid")
cols_to_log  <- setdiff(numeric_cols, dummy_cols)

train_predictor[cols_to_log] <- lapply(train_predictor[cols_to_log], log1p)
test_predictor[cols_to_log]  <- lapply(test_predictor[cols_to_log],  log1p)


if ("ID" %in% names(train_target)) {
  train_target <- train_target %>% dplyr::select(-ID)
}
if ("ID" %in% names(train_predictor)) {
  train_predictor <- train_predictor %>% dplyr::select(-ID)
  test_predictor  <- test_predictor %>% dplyr::select(-ID)
}


full_train <- cbind(train_target, train_predictor) %>% na.omit()

target_col <- names(train_target)[1]

formula_deg3 <- as.formula(paste0(target_col, " ~ .^3"))
x_train_full <- model.matrix(formula_deg3, data = full_train)[, -1]
y_train <- full_train[[target_col]]

x_test_full <- model.matrix(~ .^3, data = test_predictor)[, -1]

set.seed(123)
cv_lasso <- cv.glmnet(x_train_full, y_train, alpha = 1, standardize = TRUE)
best_lambda_lasso <- cv_lasso$lambda.min

lasso_model <- glmnet(x_train_full, y_train, alpha = 1, lambda = best_lambda_lasso, standardize = TRUE)
lasso_coefs <- coef(lasso_model)


selected_features <- rownames(lasso_coefs)[which(lasso_coefs != 0)]
selected_features <- selected_features[selected_features != "(Intercept)"]

cat("Number of selected polynomial (degree-3) features:", length(selected_features), "\n")

x_train_selected <- x_train_full[, selected_features, drop = FALSE]
x_test_selected <- x_test_full[, selected_features, drop = FALSE]


set.seed(123)
cv_ridge <- cv.glmnet(x_train_selected, y_train, alpha = 0, standardize = TRUE)
best_lambda_ridge <- cv_ridge$lambda.min

ridge_model <- glmnet(x_train_selected, y_train, alpha = 0, lambda = best_lambda_ridge, standardize = TRUE)

y_pred_train <- predict(ridge_model, s = best_lambda_ridge, newx = x_train_selected)
y_pred_train <- as.vector(y_pred_train)

residuals <- y_train - y_pred_train
n         <- length(y_train)
p         <- length(selected_features) + 1


RSS <- sum(residuals^2)
TSS <- sum((y_train - mean(y_train))^2)


R2   <- 1 - RSS / TSS
adjR2 <- 1 - (1 - R2) * (n - 1) / (n - p - 1)

AIC_val <- n * log(RSS / n) + 2 * p
BIC_val <- n * log(RSS / n) + log(n) * p

cat("Training set metrics:\n")
cat("  Adjusted R²:", round(adjR2, 4), "\n")
cat("  AIC:        ", round(AIC_val, 2), "\n")
cat("  BIC:        ", round(BIC_val, 2), "\n")

test_predictions <- predict(ridge_model, s = best_lambda_ridge, newx = x_test_selected)
predictions <- tibble(ID = 1:nrow(test_predictor), .pred = as.vector(test_predictions))
write.csv(predictions, "Mk8.csv", row.names = FALSE)
