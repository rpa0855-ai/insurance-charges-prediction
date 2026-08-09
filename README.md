
# Insurance Charges Predictor

ML-powered medical insurance charge predictor using a tuned Gradient Boosting model (R² = 0.879), with SHAP explainability and 90% prediction intervals.

## What it does

Enter a person's age, sex, BMI, number of children, smoking status, and region — the app returns a predicted insurance charge along with a 90% prediction range (not just a single number).

## Files

- `app.py` - the Streamlit app
- `requirements.txt` - dependencies
- `insurance_charge_model.pkl` - trained point-estimate model
- `insurance_charge_model_lower.pkl` - lower bound model (5th percentile)
- `insurance_charge_model_upper.pkl` - upper bound model (95th percentile)

## Run locally
