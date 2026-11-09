import pandas as pd #for data analysis
import numpy as np #for numerical operations

df = pd.read_csv('data/cumulative.csv') #reading the data


print(df.shape) #printing the shape of the data
print(df.columns) #printing the column names of the data
print(df.head()) #printing the first 5 rows of the data
df.info() #printing the information about the data
print(df.describe()) #printing the statistical summary of the data
print("No of duplicate values:", df.duplicated().sum()) #checking for duplicate values
print("No of missing values:", df.isna().sum()) #checking for missing values


print(df["koi_disposition"].value_counts()) #printing the value counts of koi_disposition column 
print("\nPredisposition:")
print(df["koi_pdisposition"].value_counts())
