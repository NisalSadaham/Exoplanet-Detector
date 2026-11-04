import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/kepler_data.csv")

df_filtered = df[df["koi_disposition"] != "CANDIDATE"].copy() #filtering the data to remove rows with koi_disposition as CANDIDATE
print(df_filtered["koi_disposition"].value_counts()) #printing the value counts of koi_disposition after filtering
df_filtered["target"] = np.where(df_filtered["koi_disposition"] == "CONFIRMED", 1, 0) #creating a new column 'target' with 1 for CONFIRMED and 0 for FALSE POSITIVE
print(df_filtered["target"].value_counts()) #printing the value counts of target column
print(df_filtered["target"].value_counts(normalize=True)* 100) #printing the value counts of target column in percentage

leakage_columns = [
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_score"
] #defining columns that might introduce target leakage

X = df_filtered.drop(columns=["koi_disposition", "target"] + leakage_columns) #features for the model by dropping the target and leakage columns
y = df_filtered["target"] #target variable for the model

print(X.shape) #printing the shape of the features
print(y.shape) #printing the shape of the target variable

leakage_columns.append("koi_pdisposition") #adding koi_pdisposition to the list of leakage columns because it had values realted to the target variable

X = df_filtered.drop(columns=["koi_disposition", "target"] + leakage_columns) #new set of features for the model by dropping the target and leakage columns
print(X.shape) #printing the shape of the new features
print(X.columns.tolist())

identifier_columns = [
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name"
] #list of identifier columns that should be dropped from the features because they do not provide any predictive power

X = X.drop(columns=identifier_columns) #dropping identifier columns from the features
print(X.shape) #printing the shape of the features after dropping identifier columns

missing_values = X.isnull().sum()

print(missing_values[missing_values > 0])

empty_columns = [
    "koi_teq_err1",
    "koi_teq_err2"
] #found out that these columns are empty and should be dropped from the features

X = X.drop(columns=empty_columns) #dropping empty columns from the features

print(X.shape)

print(X.select_dtypes(include=["object"]).columns.tolist()) #found out koi_tce_delivname is an object type column and should be dropped from the features

identifier_columns.append("koi_tce_delivname")

X = X.drop(columns=["koi_tce_delivname"]) #dropping the object type column from the features
print(X.shape) #printing the shape of the features after dropping the object type column

print("Rows with missing values:", X.isnull().any(axis=1).sum())#sum of rows with missing values in the features
print("Total rows:", len(X)) #printing the total number of rows in the features


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y) #splitting the data into training and testing sets with 80% training and 20% testing