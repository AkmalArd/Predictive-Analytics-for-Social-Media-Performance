# 📦 Libraries
library(tidyverse)
library(readr)
library(fastDummies)
library(glmnet)
library(Metrics)   # for R^2 calculation

# 📁 Load datasets
train_target <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Train_Target.csv")
train_predictor <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Training_Data_Predictors.csv")
test_predictor  <- read_csv("~/Desktop/MDA Kaggle Comp. Data/Kaggle Competition/Test_Data_Predictors.csv")

# ✅ Convert to tibbles
train_target     <- as_tibble(train_target)
train_predictor  <- as_tibble(train_predictor)
test_predictor   <- as_tibble(test_predictor)

# 🔁 Convert categorical to factor
train_predictor$Type <- as.factor(train_predictor$Type)
test_predictor$Type  <- as.factor(test_predictor$Type)

# 🧱 One-hot encode
train_predictor <- dummy_cols(train_predictor, select_columns = "Type", remove_selected_columns = TRUE)
test_predictor  <- dummy_cols(test_predictor, select_columns = "Type", remove_selected_columns = TRUE)

# Align test columns to train columns
train_predictor <- train_predictor[, -ncol(train_predictor)]
test_predictor <- test_predictor[, names(train_predictor)]

# 🧹 Remove ID column if exists
if ("ID" %in% names(train_target)) {
  train_target <- train_target[, !(names(train_target) %in% "ID"), drop = FALSE]
}
if ("ID" %in% names(train_predictor)) {
  train_predictor <- train_predictor %>% select(-ID)
  test_predictor  <- test_predictor %>% select(-ID)
}

# 🧼 Merge & clean
full_train <- cbind(train_target, train_predictor) %>% na.omit()


# 🔍 Identify top correlated variables to square
correlations <- full_train %>%
  summarise(across(all_of(numeric_cols), ~ cor(.x, full_train$Total_Interactions))) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "correlation") %>%
  arrange(desc(abs(correlation)))

# ✨ Choose top 5 correlated predictors
top_poly_vars <- head(correlations$feature, 8)

# 🧪 Add polynomial (squared) terms
for (var in top_poly_vars) {
  full_train[[paste0(var, "_sq")]] <- full_train[[var]]^3
  test_predictor[[paste0(var, "_sq")]] <- test_predictor[[var]]^3
}

# 🧠 Model matrices
X_train <- model.matrix(Total_Interactions ~ ., full_train)[, -1]
y_train <- full_train$Total_Interactions
X_test  <- as.matrix(test_predictor)

# ⚙️ Ridge Regression with Cross-validation (includes poly terms)
set.seed(123)
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)


# 🔮 Predict test set and back-transform
log_preds <- predict(ridge_model, s = ridge_model$lambda.1se, newx = X_test)
final_preds <- expm1(log_preds)

# 🧾 Add predictions
test_predictor$.pred <- pmax(0, final_preds)

# 💾 Export
predictions <- tibble(ID = 1:nrow(test_predictor), .pred = test_predictor$.pred)
write.csv(predictions, "Polynomial_Ridge_With_FeatureSelection.csv", row.names = FALSE)
