import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



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

# Evaluation of the model
mse=mean_squared_error(y_test,y_pre)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pre)
r2=r2_score(y_test,y_pre)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# save the metrics in a json files
metrics={
"Mean Squared Error  ":mse,
"Root Mean Squared Error  ":rmse,
"Mean Absolute Error: ":mae,
"R2 Score ":r2
}

# save the result of the matrics
with open(r"C:\Users\user\Desktop\Elavvo ML Internship\Student-Score-Prediction\outputs\metrics.json" , "w") as f:
          json.dump(metrics,f,indent=4)
          