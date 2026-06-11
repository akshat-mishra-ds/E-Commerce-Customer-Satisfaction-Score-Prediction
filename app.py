import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load files
model = load_model("deepcsat_ann_model.h5")

scaler = joblib.load("scaler.pkl")

feature_columns = joblib.load(
    "feature_columns.pkl"
)

st.title("DeepCSAT Prediction App")

st.write(
    "Predict Customer Satisfaction Score"
)

# Inputs

item_price = st.number_input(
    "Item Price",
    min_value=0.0
)

response_time = st.number_input(
    "Response Time (Hours)",
    min_value=0.0
)

order_age = st.number_input(
    "Order Age (Hours)",
    min_value=0.0
)

remark_length = st.number_input(
    "Remark Length",
    min_value=0
)

if st.button("Predict CSAT"):

    data = pd.DataFrame(
        np.zeros(
            (
                1,
                len(feature_columns)
            )
        ),
        columns=feature_columns
    )

    # Example mapping

    if 'Item_price' in data.columns:
        data['Item_price'] = item_price

    if 'Response_Time_Hours' in data.columns:
        data['Response_Time_Hours'] = response_time

    if 'Order_Age_Hours' in data.columns:
        data['Order_Age_Hours'] = order_age

    if 'Remark_Length' in data.columns:
        data['Remark_Length'] = remark_length

    data_scaled = scaler.transform(data)

    prediction = model.predict(
        data_scaled
    )

    predicted_class = np.argmax(
        prediction
    )

    st.success(
        f"Predicted CSAT Score: {predicted_class+1}"
    )