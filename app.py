


import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #e0f7f7;
    }
    h1, h2, h3 {
        color: #00796b;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and features
model = joblib.load("fraud_model_smote.pkl")
features = joblib.load("selected_features.pkl")

st.title("Fraud Detection App")
st.write("Enter transaction details or upload a CSV to check for fraud.")


st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

# Sidebar instructions
st.sidebar.title("Navigation")
st.sidebar.info("Use the form below to enter transaction details or upload a CSV file for batch prediction.")

with st.expander("ℹ️ About the Input Features"):
    st.markdown("""
    This model uses selected features derived from transaction metadata and anonymized variables (e.g., V1–V28) to detect fraud.  
    Key features include:
    - **scaled_amount**: Normalized transaction amount  
    - **hour**: Time of transaction  
    - **V10, V14, V17, etc.**: PCA-transformed variables capturing behavioral patterns  
    - **V14_V10_interaction**: Interaction term between V14 and V10  
    These features were selected based on their predictive power using SMOTE and LightGBM.
    """)

# Input method selection

option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    st.subheader("📝 Manual Entry – Enter Transaction Details")
    inputs = {f: st.number_input(f, value=0.0) for f in features}
    
    if st.button("Predict", key="manual_predict"):

        df = pd.DataFrame([inputs])
        df = df.reindex(columns=features, fill_value=0)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        
        st.success(f"Prediction: {' ⚠️ Fraud' if pred == 1 else '✅ Legitimate'}")
        st.info(f"Fraud Probability: {round(prob, 4)}")

        #with st.expander("See Explanation"):
          #  import shap
           # explainer = shap.TreeExplainer(model)
          #  shap_values = explainer.shap_values(df)
           # shap.initjs()
           # st.set_option('deprecation.showPyplotGlobalUse', False)
           # st.pyplot(shap.summary_plot(shap_values, df, plot_type="bar", show=False))
           # st.pyplot(shap.force_plot(explainer.expected_value, shap_values, df, matplotlib=True, show=False))
           # Detailed feature contributions
           # st.markdown("### 🔍 Feature Contributions")
           # for i in np.argsort(shap_values[1])[::-1]:
           #     st.markdown(f"**{df.columns[i]}**: {shap_values[1][i]:.4f}")

            

            # Summary in words
        st.markdown("### 📊 Summary")
        if pred == 1:
            st.warning(f"This transaction **matches known fraud patterns** with a probability of **{round(prob*100, 2)}%**.")
        else:
            st.success(f"This transaction appears **legitimate**, with only **{round((1-prob)*100, 2)}%** chance of fraud.")


elif option == "Upload CSV":

    import base64

    def download_link(df, filename, link_text):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Sample data for download
    sample_df = pd.DataFrame({
        'V14_V10_interaction': [0.5],
        'V8': [-1.2],
        'V4': [0.3],
        'hour': [14],
        'V3': [0.1],
        'V14': [-0.8],
        'V12': [0.2],
        'V19': [-0.5],
        'V6': [0.7],
        'V26': [-0.3],
        'V17': [0.6],
        'V13': [-0.4],
        'V10': [0.9],
        'V1': [0.0],
        'V15': [-0.2],
        'V22': [0.1],
        'V7': [0.3],
        'scaled_amount': [0.5],
        'V2': [-0.1],
        'V20': [0.2],
        'V16': [0.4]
    })

    st.markdown("### 📥 Sample CSV for Testing")
    download_link(sample_df, "sample.csv", "Click here to download sample CSV")

    
    st.subheader("📁 Upload CSV File")
    file = st.file_uploader("Upload your transaction file", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        df = df.reindex(columns=features, fill_value=0)
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        st.subheader("Prediction Results")

        # After prediction
        df["Prediction"] = ["⚠️ Fraud" if p == 1 else "✅ Legitimate" for p in preds]
        df["Fraud Probability"] = probs

        has_fraud = any(p == 1 for p in preds)
        
        # Show alert only if fraud exists
        if has_fraud:
            st.error("⚠️ Some transactions match known fraud patterns. Please review the results below.")
            st.markdown('<button style="background-color:red;color:white;padding:10px;border:none;border-radius:5px;">Fraud Detected</button>', unsafe_allow_html=True)    
        else:
            st.success("✅ All transactions appear legitimate.")
        
        
        st.markdown("### 📊 Summary")
        
        for i in range(len(df)):
            label = df.loc[i, "Prediction"]
            prob = df.loc[i, "Fraud Probability"]
            print(label, prob)
    
            if label != "Fraud":
                st.warning(f"Transaction {i+1} matches known fraud patterns with a probability of **{round(prob*100, 2)}%**.")
            else:
                st.success(f"Transaction {i+1} appears legitimate with only **{round((1-prob)*100, 2)}%** chance of fraud.")

        # Highlight frauds in red
        def highlight_fraud(row):
         return ['background-color: red' if row["Prediction"] == "Fraud" else '' for _ in row]

        

        styled_df = df.style.apply(highlight_fraud, axis=1)
        st.dataframe(styled_df)

        st.download_button("Download Results", df.to_csv(index=False), "fraud_results.csv")

