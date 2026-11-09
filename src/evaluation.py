
from model_training import trained_pipeline #imported the trained pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score #imported the necessary metrics from sklearn to evaluate the model
from data_preprocessing import x_test, y_test,x_train #imported the testing features and target variable from data_preprocessing.py
from model_training import best_model #imported the best model from model_training.py
import pandas as pd #imported pandas to create a dataframe for the evaluation metrics

y_hat = trained_pipeline.predict(x_test) #predicting the target variable on the testing features using the trained pipeline
accuracy = accuracy_score(y_test, y_hat) #calculating the accuracy of the model
precision = precision_score(y_test, y_hat) #calculating the precision of the model
recall = recall_score(y_test, y_hat) #calculating the recall of the model
f1 = f1_score(y_test, y_hat) #calculating the F1 score of the model
confusion_mat = confusion_matrix(y_test, y_hat) #calculating the confusion matrix of the model

print("Accuracy:", accuracy) #printing the accuracy of the model
print("Precision:", precision) #printing the precision of the model
print("Recall:", recall) #printing the recall of the model
print("F1 Score:", f1) #printing the F1 score of the model
print("Confusion Matrix:\n", confusion_mat) #printing the confusion matrix of the model


best_y_hat = best_model.predict(x_test) #predicting the target variable on the testing features using the best model from grid search
best_accuracy = accuracy_score(y_test, best_y_hat) #calculating the accuracy of the best model
best_precision = precision_score(y_test, best_y_hat) #calculating the precision of the best model
best_recall = recall_score(y_test, best_y_hat) #calculating the recall of the best model
best_f1 = f1_score(y_test, best_y_hat) #calculating the F1 score of the best model

print("Best predicted values:", best_y_hat) #printing the predicted values of the best model
print("Best predicited probabilities:", best_model.predict_proba(x_test)[:, 1]) #printing the predicted probabilities of the positive class of the best model
print("Best Model Accuracy:", best_accuracy) #printing the accuracy of the best model
print("Best Model Precision:", best_precision) #printing the precision of the best model
print("Best Model Recall:", best_recall) #printing the recall of the best model
print("Best Model F1 Score:", best_f1) #printing the F1 score of the best model

print("Best Model Confusion Matrix:\n", confusion_matrix(y_test, best_y_hat)) #printing the confusion matrix of the best model

best_model_params = best_model.named_steps['model']

print("coefficients:", best_model_params.coef_) #printing the coefficients of the best model
print("intercept:", best_model_params.intercept_) #printing the intercept of the best model

print(x_train.columns) #printing the columns of the training features

df = pd.DataFrame({"columns": x_train.columns, "coefficients": best_model_params.coef_[0]}) #creating a dataframe with the columns and coefficients of the best model
print(df) #printing the dataframe with the columns and coefficients of the best model