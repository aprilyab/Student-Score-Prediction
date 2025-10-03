# import packages and libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


# train the model
model=LinearRegression()
model.fit(x_train,y_train)

# make prediction from the model by using x_testPrimitive Data Structures:
y_pre=model.predict(x_test.reshape())

# save the model for fature uage
joblib.dump(model,r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\outputs\models\Student-Performance-Predictor-Model")

