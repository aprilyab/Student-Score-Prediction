# Student Score Prediction Project

## Project Overview

This project aims to **predict students’ exam scores  based on their study hours (`studytime`)** using linear regression. The analysis includes **data cleaning, preprocessing, model training, evaluation, and visualization**.

This project demonstrates the **end-to-end workflow of a predictive modeling pipeline**, including exploratory data analysis, feature engineering, and model evaluation.

---

**Live App:** [Student Score Predictor](https://student-score-prediction-wxifwpnjbdrgujsffwd9p7.streamlit.app/)

## Table of Contents

- [Student Score Prediction Project](#student-score-prediction-project)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
  - [Model Training](#model-training)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Visualizations](#visualizations)
  - [References](#references)

---

## Dataset

* **Recommended Dataset:** Student Performance Factors (Kaggle)
* **File Used:** `student-mat.csv`
* **Target Variable:** `Exam_Score` → Final grade in the course
* **Feature Used:** `Hours_Studied` → Weekly study time (1–4)

---

## Project Structure

```
student-score-prediction/
│
├── data/                   # Original dataset (CSV files)
│   └──raw/StudentPerformanceFactors.csv
│   |__processed/cleaned_Student_Performance_Factors.csv
|
├── notebooks/              # Jupyter notebooks
│   └── Explanatory_Data_Analysis.ipynb
│
├── outputs/                # Saved models, cleaned datasets, and plots
│   ├── metrics.json
│   ├── models/student_score_model.pkl
│   └── figurs/True-vs-Predicted.png
│
|__ .venv
|
├── src/                    # Python scripts for modular code
│   ├── data_cleaning_&_processing.py
│   └── model_training.py
|   |__ Visualize predictions.py
|   |__ Model_evaluation.py
│
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Requirements

Install the following Python libraries using:

```bash
pip install -r requirements.txt
```

**Python Libraries:**

* `pandas` → Data manipulation
* `numpy` → Numerical computations
* `matplotlib` → Visualization
* `seaborn` → Advanced visualization
* `scikit-learn` → Model training & evaluation
* `joblib` → Save and load trained models

---

## Data Cleaning and Preprocessing

Steps performed to prepare data for modeling:

1. **Select relevant columns:** `studytime` and `G3`
2. **Handle missing values:** Remove rows with missing data
3. **Map studytime categories:** Convert ordinal categories (1–4) into approximate hours
4. **Check for outliers:** Ensure `G3` scores are between 0–20
5. **Split dataset:** Training set (80%) and testing set (20%)
6. **Optional scaling:** For models that require it (not necessary for linear regression)

---

## Model Training

* **Model Used:** Linear Regression
* **Feature:** `Hours_Studied`
* **Target:** `Exam_Score`
* **Training Steps:**

  1. Train the model on the training set
  2. Make predictions on the testing set

---

## Evaluation Metrics

The model was evaluated using the following metrics:

```json
{
    "Mean Squared Error": 12.9241,
    "Root Mean Squared Error": 3.5950,
    "Mean Absolute Error": 2.5495,
    "R2 Score": 0.1744
}
```

**Interpretation:**

* The model explains about 17% of the variance in final exam scores based on study hours alone.
* RMSE and MAE indicate average prediction errors around 3.6 and 2.55 points respectively.

---

## Visualizations

1. **Scatter Plot with Regression Line** → True vs Predicted scores


Plots are saved in the `outputs/figures` folder as PNG files for reports or presentations.

1. Install requirements:


pip install -r requirements.txt

3. Run the notebook or Python scripts in `src/` to reproduce the analysis
4. All cleaned data, predictions, and plots are saved in the `outputs/` folder

---

## References

* [Kaggle Dataset: Student Performance Factors](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
* [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
* [Seaborn Documentation](https://seaborn.pydata.org/)

---

**Notes:**

* The project can be extended by including **additional features**  for improved prediction accuracy.
* This README serves as a **complete guide** for project reproducibility, analysis, and visualization.

** Author:** Henok Yoseph
Email:henokapril@gmail.com
gitub: https://github.com/aprilyab 

