import pandas as pd #for data analysis
import matplotlib.pyplot as plt #for visualization 
import numpy as np #for numerical operations

df = pd.read_csv('data/cumulative.csv') #reading the data


#print(df.shape) #printing the shape of the data
#print(df.columns) #printing the column names of the data
#print(df.head()) #printing the first 5 rows of the data
#print(df.info()) #printing the information about the data
#print(df.describe()) #printing the statistical summary of the data
#print("No of missing values:", df.isnull().sum()) #checking for missing values
#print("No of duplicate values:", df.duplicated().sum()) #checking for duplicate values
#print("No of missing values:", df.isna().sum()) #checking for missing values


#print(df["koi_disposition"].value_counts())

#print("\nPredisposition:")
#print(df["koi_pdisposition"].value_counts())

df_filtered = df[df["koi_disposition"] != "CANDIDATE"].copy() #filtering the data to remove rows with koi_disposition as CANDIDATE
print(df_filtered["koi_disposition"].value_counts()) #printing the value counts of koi_disposition after filtering
df_filtered["target"] = np.where(df_filtered["koi_disposition"] == "CONFIRMED", 1, 0) #creating a new column 'target' with 1 for CONFIRMED and 0 for FALSE POSITIVE
print(df_filtered["target"].value_counts()) #printing the value counts of target column
print(df_filtered["target"].value_counts(normalize=True)* 100) #printing the value counts of target column in percentage