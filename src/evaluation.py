from model_training import trained_pipeline #imported the trained pipeline 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score #imported the necessary metrics from sklearn to evaluate the model
from data_preprocessing import x_test, y_test #imported the testing features and target variable from data_preprocessing.py

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