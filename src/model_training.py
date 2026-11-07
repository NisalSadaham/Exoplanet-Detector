from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_preprocessing import X_train_scaled, X_test_scaled, y_train, y_test,x_train, x_test
#print(X_train_scaled.shape) #printing the shape of the scaled training features
#print(X_test_scaled.shape) #printing the shape of the scaled testing features
from sklearn.linear_model import LogisticRegression #importing logistic regression model from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix #importing metrics to evaluate the model
from sklearn.dummy import DummyClassifier #importing dummy classifier from sklearn to compare the performance of the model with a baseline model
from sklearn.model_selection import StratifiedKFold #importing stratified k-fold cross-validation and cross-validation score from sklearn
from sklearn.pipeline import Pipeline #importing pipeline from sklearn to create a pipeline for the model
model = LogisticRegression(random_state=42) #creating an instance of logistic regression model
dummy_model = DummyClassifier(strategy='most_frequent', random_state=42) #creating an instance of dummy classifier
from sklearn.model_selection import cross_val_score #importing cross-validation score from sklearn to evaluate the model using cross-validation

model.fit(X_train_scaled, y_train) #fitting the model on the scaled training features and target variable
dummy_model.fit(X_train_scaled, y_train) #fitting the dummy model on the scaled training features and target variable

y_pred = model.predict(X_test_scaled) #predicting the target variable on the scaled testing features
y_pred_proba = model.predict_proba(X_test_scaled) #predicting the probability of the positive class on the scaled testing features

#print("Predicted class:", y_pred) #printing the predicted class
#print("Predicted probability of positive class:", y_pred_proba[:, 1]) #printing the predicted probability of the positive class

dummy_y_pred = dummy_model.predict(X_test_scaled) #predicting the target variable on the scaled testing features using the dummy model

accuracy = accuracy_score(y_test, y_pred) #calculating the accuracy of the model
precision = precision_score(y_test, y_pred) #calculating the precision of the model
recall = recall_score(y_test, y_pred) #calculating the recall of the model
f1 = f1_score(y_test, y_pred) #calculating the F1 score of the model

#print("Accuracy:", accuracy)
#print("Precision:", precision)
#print("Recall:", recall)
#print("F1 Score:", f1)
#print("Confusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

dummy_accuracy = accuracy_score(y_test, dummy_y_pred) #calculating the accuracy of the dummy model
#print("Dummy Model Accuracy:", dummy_accuracy)
dummy_precision = precision_score(y_test, dummy_y_pred) #calculating the precision of the dummy model
#print("Dummy Model Precision:", dummy_precision)
dummy_recall = recall_score(y_test, dummy_y_pred) #calculating the recall of the dummy model
#print("Dummy Model Recall:", dummy_recall)
dummy_f1 = f1_score(y_test, dummy_y_pred) #calculating the F1 score of the dummy model
#print("Dummy Model F1 Score:", dummy_f1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #creating an instance of stratified k-fold cross-validation with 5 splits

pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')), #adding simple imputer to the pipeline to fill missing values in features
    ('scaler',StandardScaler()), #adding standard scaler to the pipeline to scale the features
    ('model', LogisticRegression(random_state=42)) #adding the logistic regression model to the pipeline
])

cross_val_scores = cross_val_score(pipeline,X=x_train,y=y_train,cv=skf,scoring='recall') #evaluating the model using cross-validation with stratified k-fold cross-validation and recall as the scoring metric
#print("Cross-validation scores:", cross_val_scores) #printing the cross-validation scores
#print("Mean cross-validation score:", cross_val_scores.mean()) #printing the mean of the cross-validation scores
#print("Standard deviation of cross-validation scores:", cross_val_scores.std()) #printing the standard deviation of the cross-validation scores

cross_val_scores_2 = cross_val_score(pipeline,X=x_train,y=y_train,cv=skf,scoring='precision') #evaluating the model using cross-validation with stratified k-fold cross-validation and precision as the scoring metric
#print("Cross-validation scores (Precision):", cross_val_scores_2) #printing the cross-validation scores
#print("Mean cross-validation score (Precision):", cross_val_scores_2.mean()) #printing the mean of the cross-validation scores
#print("Standard deviation of cross-validation scores (Precision):", cross_val_scores_2.std()) #printing the standard deviation of the cross-validation scores

cross_val_scores_3 = cross_val_score(pipeline,X=x_train,y=y_train,cv=skf,scoring='f1') #evaluating the model using cross-validation with stratified k-fold cross-validation and f1 score as the scoring metric
#print("Cross-validation scores (F1):", cross_val_scores_3) #printing the cross-validation scores
#print("Mean cross-validation score (F1):", cross_val_scores_3.mean()) # printing the mean of the cross-validation scores
#print("Standard deviation of cross-validation scores (F1):", cross_val_scores_3.std()) #printing the standard deviation of the cross-validation scores

trained_pipeline = pipeline.fit(x_train, y_train) #fitting the pipeline on the training features and target variable   
