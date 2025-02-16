# Load required libraries
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(readr)
library(rmarkdown)

# Load the training and testing datasets
train_path <- "pml-training.csv"
test_path <- "pml-testing.csv"

if (file.exists(train_path) & file.exists(test_path)) {
  training <- read_csv(train_path)
  testing <- read_csv(test_path)
} else {
  stop("Training or testing dataset not found. Please check file paths.")
}

# Data preprocessing
na_threshold <- 0.95
na_counts <- colSums(is.na(training)) / nrow(training)
training <- training[, na_counts < na_threshold]

# Ensure testing dataset has the same features as training (excluding 'classe')
common_features <- setdiff(names(training), "classe")
testing <- testing[, common_features, drop = FALSE]

# Check if testing dataset is empty
debug_testing <- nrow(testing) == 0
if (debug_testing) {
  warning("Testing dataset is empty. Predictions will be skipped.")
}

# Remove irrelevant columns
training <- training[, -c(1:7)]
training$classe <- as.factor(training$classe)

# Split training data
set.seed(456)
trainIndex <- createDataPartition(training$classe, p = 0.7, list = FALSE)
trainData <- training[trainIndex, ]
validData <- training[-trainIndex, ]

# Train a Random Forest model
set.seed(456)
rf_model <- randomForest(classe ~ ., data = trainData, ntree = 150)

# Validate model
rf_predictions <- predict(rf_model, validData)
conf_matrix <- confusionMatrix(rf_predictions, validData$classe)
print(conf_matrix)

# Feature importance
importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df %>% arrange(desc(MeanDecreaseGini))

# Plot feature importance
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance")

# Apply model to test dataset
if (!debug_testing) {
  final_predictions <- predict(rf_model, testing)
  print(head(final_predictions))  # Debugging output
} else {
  final_predictions <- NULL
}

# Save predictions only if they exist
if (!is.null(final_predictions) && length(final_predictions) > 0) {
  prediction_files <- function(predictions) {
    for (i in seq_along(predictions)) {
      filename <- paste0("prediction_result_", i, ".txt")
      print(paste("Attempting to write file:", filename))  # Debugging output
      tryCatch(
        {
          write.table(predictions[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
          message("Successfully written: ", filename)
        },
        error = function(e) {
          message("Error writing file: ", filename, " - ", e$message)
        }
      )
    }
  }
  prediction_files(final_predictions)
} else {
  warning("final_predictions is empty. Skipping file saving.")
}

# Save model if it exists
if (exists("rf_model")) {
  saveRDS(rf_model, "rf_model.rds")
} else {
  warning("Model 'rf_model' not found. Skipping save.")
}

# Generate R Markdown report
if (file.exists("prediction report.Rmd")) {
  render("prediction report.Rmd", output_format = "html_document")
} else {
  warning("File 'prediction report.Rmd' not found in", getwd(), ". Listing available files:")
  print(list.files())
}
