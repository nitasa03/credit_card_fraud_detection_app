
import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("fraud_model_smote.pkl")
features = joblib.load("selected_features.pkl")

st.title("Fraud Detection App")
st.write("Enter transaction details or upload a CSV to check for fraud.")

option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    st.subheader(" Enter Transaction Details")
    inputs = {f: st.number_input(f, value=0.0) for f in features}
    
    if st.button("Predict"):
        df = pd.DataFrame([inputs])
        df = df.reindex(columns=features, fill_value=0)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        
        st.success(f"Prediction: {' Fraud' if pred == 1 else ' Legit'}")
        st.info(f"Fraud Probability: {round(prob, 4)}")

elif option == "Upload CSV":
    st.subheader("Upload CSV File")
    file = st.file_uploader("Upload your transaction file", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        df = df.reindex(columns=features, fill_value=0)
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        
        df["Prediction"] = preds
        df["Fraud Probability"] = probs
        st.dataframe(df)

        st.download_button("Download Results", df.to_csv(index=False), "fraud_results.csv")
