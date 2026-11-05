import pandas as pd
import numpy as np
from data_preprocessing import X_train_scaled, X_test_scaled, y_train, y_test
#print(X_train_scaled.shape) #printing the shape of the scaled training features
#print(X_test_scaled.shape) #printing the shape of the scaled testing features
from sklearn.linear_model import LogisticRegression #importing logistic regression model from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix #importing metrics to evaluate the model

model = LogisticRegression(random_state=42) #creating an instance of logistic regression model

model.fit(X_train_scaled, y_train) #fitting the model on the scaled training features and target variable

y_pred = model.predict(X_test_scaled) #predicting the target variable on the scaled testing features
y_pred_proba = model.predict_proba(X_test_scaled) #predicting the probability of the positive class on the scaled testing features

print("Predicted class:", y_pred) #printing the predicted class
print("Predicted probability of positive class:", y_pred_proba[:, 1]) #printing the predicted probability of the positive class


accuracy = accuracy_score(y_test, y_pred) #calculating the accuracy of the model
precision = precision_score(y_test, y_pred) #calculating the precision of the model
recall = recall_score(y_test, y_pred) #calculating the recall of the model
f1 = f1_score(y_test, y_pred) #calculating the F1 score of the model

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))