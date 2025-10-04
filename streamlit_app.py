""" from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
import pandas as pd

model=joblib.load("outputs/models/Student-Performance-Predictor-Model")
scaler=StandardScaler()

st.title("Student Performance Predictor App")

st.image("data/raw/student performance.png",use_container_width=True)
st.write("enter your study time")

study_time=st.number_input("study time",min_value=1,max_value=44,value=20)
df=pd.DataFrame([[study_time]], columns=["study_time"])

x=scaler.fit_transform(df)

if st.button("predict the exam score"):
    score=model.predict(x)
    st.write(f"predicted score: {score}")
    """

import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load Model
# ===============================


model = joblib.load("outputs/models/Student-Performance-Predictor-Model")

# ===============================
# Streamlit App UI
# ===============================


st.title("🎓 Student Performance Predictor")
st.image("data/raw/student_performance.png",width="strech")
st.write("This app predicts a student's exam score based on their **study time** (hours per week).")

# Sidebar inputs
st.sidebar.header(" Enter Input Data")
study_time = st.sidebar.number_input(
    "Study Time (hours per week)",
    min_value=1,
    max_value=44,
    value=20,
    step=1
)

# Put input into DataFrame
input_df = pd.DataFrame([[study_time]], columns=["study_time"])

# Prediction button
if st.sidebar.button(" Predict Score"):
    prediction = model.predict(input_df)
    predicted_score = round(prediction[0],2)

    st.subheader("📊 Prediction Result")
    st.success(f"✅ The predicted exam score is **{predicted_score}**")

    # Display metrics
    st.metric(label="Study Time (hours/week)", value=study_time)
    st.metric(label="Predicted Exam Score", value=predicted_score)

    # Extra visualization
    st.progress(min(predicted_score, 100) / 100)
    st.write("🔵 Progress bar shows predicted performance towards 100.")

# Footer
st.markdown("---")
st.caption("Made with  using Streamlit")
