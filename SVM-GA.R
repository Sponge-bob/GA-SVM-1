#Importing caret library
library(caret)
library(doParallel)

#Importing dataset
tea_df <- read.csv("D:/Kuliah/SKRIPSWEET/SKRIPSI/Working space/Data sampel/TOR-C/Final data/All/hasil/Results FE MAX.csv", sep = ',', header = TRUE)
str(tea_df)

#Data slicing
set.seed(11)
intrain <- createDataPartition(y = tea_df$nama, p= 0.7, list = FALSE)
training <- tea_df[intrain,]
testing <- tea_df[-intrain,]

#Checking dimensions of data frame
dim(training);dim(testing);

#Preprocessing and Training
#Checking missing values
anyNA(tea_df)

#Dataset summary
summary(tea_df)

#Converting target variable to factor variable
training[["nama"]] = factor(training[["nama"]])

#Training SVM Model
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(22)

svm_classification <- train(nama ~., data = training, method = "svmLinear",
                            trControl=trctrl,
                            preProcess = c("center", "scale"),
                            tuneLength = 10)

#SVM Model result
svm_classification

#Test Set Prediction
test_pred <- predict(svm_classification, newdata = testing)
test_pred

#SVM Accuracy
confusionMatrix(test_pred, testing$nama)

#Multicores
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Define control function
ga_ctrl <- gafsControl(functions = caretGA,  
                       method = "cv",
                       number = 10)

# Genetic Algorithm feature selection
set.seed(10)
system.time(ga_obj <- gafs(x=tea_df[,c(1:8)], 
                           y=tea_df[,c(12)],
                           iters = 50,   # normally much higher (100+)
                           gafsControl = ga_ctrl))
stopCluster(cl)
ga_obj

# Optimal variables
ga_obj$optVariables

#SVM with the selected features
test_pred_SF <- predict(ga_obj, newdata = testing)
test_pred_SF

#SVM after Feature Selection
confusionMatrix(test_pred_SF, testing$nama)

