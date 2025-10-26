import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model and Scaler (Cached) ---
# Use st.cache_resource to load these only once at the start of the app
@st.cache_resource
def load_model_and_scaler():
    """
    Loads the saved model and scaler from disk.
    Returns (None, None) if files are not found.
    """
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model and scaler loaded successfully for Streamlit app.")
        return model, scaler
    except FileNotFoundError:
        st.error("--- FATAL ERROR ---")
        st.error("Error: 'fraud_model.pkl' or 'scaler.pkl' not found.")
        st.error("Please run the 'fraud_detection_save_model.py' script first to train and save the files.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the files: {e}")
        return None, None

# --- 2. Prediction Function ---
# This is the same function from your 'run_fraud_prediction.py' script
def predict_transaction(model, scaler, transaction_data):
    """
    Predicts if a single transaction is fraudulent or not.

    Args:
        model: The trained model object.
        scaler: The fitted scaler object.
        transaction_data (dict): A dictionary containing all V1-V28 features
                                 and the original 'Amount'.
    """
    try:
        df = pd.DataFrame([transaction_data])
        amount = df['Amount'].values.reshape(-1, 1)
        
        # --- Preprocessing ---
        scaled_amount = scaler.transform(amount)
        df['Scaled_Amount'] = scaled_amount
        features_for_model = [f'V{i}' for i in range(1, 29)] + ['Scaled_Amount']
        final_data = df[features_for_model]
        
        # --- Prediction ---
        prediction_proba = model.predict_proba(final_data)[0][1] # Prob of class 1
        prediction = model.predict(final_data)[0]
        
        return prediction, prediction_proba

    except KeyError as e:
        st.error(f"Error: Missing feature {e} in transaction data.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# --- 3. Streamlit App UI ---
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Real-Time Fraud Detection")
st.write("This app uses a trained Random Forest model to predict if a transaction is fraudulent.")

# Load the model and scaler
model, scaler = load_model_and_scaler()

if model and scaler:
    st.sidebar.header("Check a New Transaction")
    
    # --- Input Widget ---
    # We only ask for Amount, as V1-V28 are not human-readable
    amount = st.sidebar.number_input("Transaction Amount ($)", 
                                     min_value=0.0, 
                                     value=100.00, 
                                     step=10.0,
                                     format="%.2f")

    # --- Prediction Button ---
    if st.sidebar.button("Check Transaction", type="primary"):
        
        # Create the dummy V-features (all zeros)
        v_features = {f'V{i}': 0 for i in range(1, 29)}
        
        # Combine with the user-provided Amount
        sample_tx = v_features.copy()
        sample_tx['Amount'] = amount
        
        # Run the prediction
        pred, proba = predict_transaction(model, scaler, sample_tx)
        
        if pred is not None:
            st.header("Prediction Result")
            
            # Show a colored box based on the result
            if pred == 1:
                st.error("Prediction: FRAUDULENT TRANSACTION", icon="🚨")
            else:
                st.success("Prediction: NORMAL TRANSACTION", icon="✅")
            
            # Show the probability
            st.metric(label="Probability of Fraud", 
                      value=f"{proba * 100:.2f}%")
            
            st.subheader("Transaction Details")
            st.json({"Input Amount": f"${amount:.2f}",
                     "V1-V28 Features": "Using demo values (all zeros)"
                    })

    # Explanation of the V-features
    with st.sidebar.expander("ℹ️ About V1-V28 Features"):
        st.info(
            "In the real dataset, V1-V28 are anonymized features from the "
            "payment processor. Since we can't ask for them, "
            "this demo uses '0' for all 28 V-features and focuses on the 'Amount'."
        )

else:
    st.warning("Model files not loaded. Please ensure 'fraud_model.pkl' and 'scaler.pkl' are in the same folder.")
