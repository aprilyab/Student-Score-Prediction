import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# load the trained model
model=joblib.load(r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\outputs\models\Student-Performance-Predictor-Model")

# load the processed and cleaned data
df=pd.read_csv(r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\data\processed\cleaned_Student_Performance_Factors.csv")

# identify the target and predictor features
x=df["Hours_Studied"].to_numpy()
y=df['Exam_Score'].to_numpy()

# split the dataset into test and train datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# perform standardization
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train.reshape(-1,1))
x_test=scaler.transform(x_test.reshape(-1,1))

# make prediction from the 
y_pre=model.predict(x_test.reshape(-1,1))

# visusalize the prediction and the actual value vs the study time
plt.figure(figsize=(10,12))
sns.scatterplot(x=x_test.reshape(-1,1).squeeze(),y=y_test,label="True value")
sns.scatterplot(x=x_test.reshape(-1,1).squeeze(),y=y_pre,label="Predicted value")
plt.title("True VS Predicted Exam Score")
plt.xlabel("Study Time")
plt.ylabel("Exam Score")
plt.savefig(r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\outputs\figures\True-vs-Predicted.png")
plt.show()





          