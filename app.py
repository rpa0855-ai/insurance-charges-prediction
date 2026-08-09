import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Charges Predictor", page_icon="💰", layout="centered")

model = joblib.load("insurance_charge_model.pkl")
lower_model = joblib.load("insurance_charge_model_lower.pkl")
upper_model = joblib.load("insurance_charge_model_upper.pkl")

bins = [0, 18.5, 25, 30, float("inf")]
labels = ["underweight", "normal", "overweight", "obese"]

st.title("Insurance Charges Predictor")
st.write("Enter your details below to estimate medical insurance charges.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Charges", type="primary"):
    bmi_category = pd.cut([bmi], bins=bins, labels=labels)[0]
    smoker_bmi = bmi if smoker == "yes" else 0

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "bmi_category": bmi_category,
        "smoker_bmi": smoker_bmi
    }])

    prediction = model.predict(input_df)[0]
    low = lower_model.predict(input_df)[0]
    high = upper_model.predict(input_df)[0]

    st.divider()
    st.metric("Estimated Annual Charges", f"${prediction:,.2f}")
    st.caption(f"90% prediction interval: ${low:,.2f} - ${high:,.2f}")

    if smoker == "yes":
        st.warning("Smoking is the single biggest driver of higher charges in this model.")
    if bmi >= 30:
        st.info(f"BMI category: obese. Combined with smoking status, this has a strong effect on the estimate.")

st.divider()
st.caption("Model: tuned Gradient Boosting pipeline trained on the Medical Cost Personal Dataset.")
