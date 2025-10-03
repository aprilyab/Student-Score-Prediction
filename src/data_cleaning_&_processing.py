import pandas as pd 
import numpy as np

# load the raw data
df=pd.read_csv(r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\data\raw\StudentPerformanceFactors.csv")

# identify the numbeer of null rows
print(df.isnull().sum())

# remove rows with null value
df=df.dropna()

num_cols=df.select_dtypes(include=["float64","int64"])
cat_cols=df.select_dtypes(include=["object","category"])

# Keep only relevant columns (feature + target)
df = df[['Hours_Studied', 'Exam_Score']]

# Drop rows with missing values in 'studytime' or 'G3'
df = df.dropna(subset=['Hours_Studied', 'Exam_Score'])

# Ensure 'Hours_Studied' is numeric
df.Hours_Studied=df.Hours_Studied.astype(int)

# check th outlier in the target variable 'Exam_Score'
print(f"min value of the 'Exam_Score' {df['Exam_Score'].min()} ")
print(f"max value of the 'Exam_Score' {df['Exam_Score'].max()} ")

df=df[(df['Exam_Score']>=0)  &  (df['Exam_Score']<=100) ]
print(df)

df.to_csv(r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\data\processed\cleaned_Student_Performance_Factors.csv")

print(f"min value of the 'Exam_Score' {df['Hours_Studied'].min()} ")
print(f"max value of the 'Exam_Score' {df['Hours_Studied'].max()} ")