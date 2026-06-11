# DeepCSAT: E-Commerce Customer Satisfaction Score Prediction
---
## Project Overview

Customer Satisfaction (CSAT) is one of the most important performance indicators for e-commerce businesses. Understanding customer satisfaction helps organizations improve service quality, reduce churn, increase customer retention, and enhance overall customer experience.

This project focuses on predicting Customer Satisfaction (CSAT) Scores using Artificial Neural Networks (ANN) based on customer support interaction data collected from an e-commerce platform.

The solution leverages data preprocessing, feature engineering, exploratory data analysis, hypothesis testing, and deep learning techniques to predict customer satisfaction levels from customer support interactions.

---

## Problem Statement

Customer satisfaction directly impacts customer loyalty and business growth. However, manually evaluating customer feedback and service interactions can be time-consuming and inefficient.

The goal of this project is to build a Deep Learning based prediction system capable of forecasting customer satisfaction scores using historical customer support interaction data.

The developed model can help businesses:

* Identify dissatisfied customers proactively.
* Improve support quality and response efficiency.
* Optimize operational performance.
* Support data-driven decision making.
* Enhance customer experience.

---

## Business Objective

The objective of this project is to predict CSAT Scores using customer support interaction data and generate actionable insights that help businesses improve customer satisfaction and service performance.

---

## Dataset Information

The dataset contains customer support interaction records from an e-commerce platform.

### Features Included

* Channel Name
* Category
* Sub-Category
* Customer Remarks
* Customer City
* Product Category
* Item Price
* Connected Handling Time
* Agent Name
* Supervisor
* Manager
* Tenure Bucket
* Agent Shift
* Order Date
* Issue Reported Time
* Issue Responded Time
* Survey Response Date
* CSAT Score (Target Variable)

### Dataset Size

* Total Records: 85,907
* Target Variable: CSAT Score
* Problem Type: Multi-Class Classification

---

## Project Workflow

### 1. Data Understanding

* Dataset exploration
* Feature analysis
* Missing value analysis
* Duplicate record analysis

### 2. Data Wrangling

* Missing value treatment
* Data quality checks
* Identifier column removal

### 3. Exploratory Data Analysis

* CSAT Score distribution
* Channel-wise analysis
* Product category analysis
* Agent performance analysis
* Response time analysis
* Correlation analysis

### 4. Hypothesis Testing

Business-focused statistical hypothesis testing was performed to identify relationships between:

* Support channels and CSAT
* Agent tenure and CSAT
* Product category and CSAT

### 5. Feature Engineering

The following engineered features were created:

* Response_Time_Hours
* Survey_Delay_Hours
* Order_Age_Hours
* Order_Hour
* Order_Day
* Order_Month
* Missing Value Indicators
* Remark_Length

### 6. Data Preprocessing

* Missing value imputation
* One-Hot Encoding using `pd.get_dummies()`
* Feature Scaling using StandardScaler
* Train-Test Split

### 7. Deep Learning Model Development

An Artificial Neural Network (ANN) was developed using TensorFlow and Keras.

Model Components:

* Dense Layers
* Batch Normalization
* Dropout Layers
* Early Stopping
* Learning Rate Reduction

---

## Model Performance

### Evaluation Metrics

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 70.04% |
| Precision | 55.70% |
| Recall    | 70.00% |
| F1 Score  | 59.93% |

The ANN model demonstrated good predictive capability and generalization performance on unseen customer interaction data.

---

## Key Insights

* Customer satisfaction is highly influenced by support interaction characteristics.
* Support channel selection affects customer ratings.
* Product categories contribute to satisfaction variations.
* Response time and handling efficiency influence customer experience.
* Agent experience impacts customer satisfaction outcomes.

---

## Local Deployment

The trained ANN model was deployed locally using Streamlit.

### Deployment Steps

1. Saved trained ANN model (`deepcsat_ann_model.h5`)
2. Saved StandardScaler (`scaler.pkl`)
3. Saved feature column structure (`feature_columns.pkl`)
4. Developed Streamlit application (`app.py`)
5. Loaded saved artifacts into Streamlit
6. Generated real-time CSAT predictions

### Run Application

```bash
streamlit run app.py
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow
* Keras
* Joblib
* Streamlit

---

## Repository Structure

```text
DeepCSAT_Project/
│
├── DeepCSAT_E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb
├── app.py
├── deepcsat_ann_model.h5
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── dataset.csv
```

---

## Future Improvements

* Implement SMOTE for imbalance handling.
* Experiment with XGBoost and LightGBM.
* Use advanced NLP models for customer remarks.
* Perform hyperparameter optimization.
* Deploy on cloud platforms such as AWS or Streamlit Cloud.
* Integrate real-time customer interaction streams.

---

## Conclusion

This project successfully developed a Deep Learning based Customer Satisfaction Prediction System using Artificial Neural Networks.

The model achieved an accuracy of 70.04% and demonstrated the ability to predict customer satisfaction using customer support interaction data.

The solution can assist organizations in improving customer service quality, identifying satisfaction trends, and supporting data-driven business decisions.

---

### Author

**Akshat Mishra**

Master's Program in Data Science
AlmaBetter × Woolf University
