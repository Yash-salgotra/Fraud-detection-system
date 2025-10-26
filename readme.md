Credit Card Fraud Detection App

This project uses Python, scikit-learn, and Streamlit to build a complete machine learning application for detecting credit card fraud.

The workflow is divided into two main parts:

1.Model Training (fraud_detection_save_model.py): This script loads a transaction dataset, preprocesses it, handles the severe class imbalance using SMOTE, trains a Random Forest model, and saves the trained model and data scaler to disk.

2.Streamlit App (fraud_app.py): This script loads the saved model and provides a simple web-based UI where a user can input a transaction amount to get a real-time fraud prediction.

Files in this Project

·creditcard.csv: (Required) The dataset containing transaction data. You must download this from Kaggle.

·fraud_detection_save_model.py: The Python script to train and save the model.

·fraud_app.py: The Python script for the Streamlit user interface.

·run_fraud_prediction.py: (Optional) A command-line script to test the saved model.

·fraud_model.pkl: (Output) The saved, trained Random Forest model.

·scaler.pkl: (Output) The saved StandardScaler used for the 'Amount' feature.

·readme.md: This file.

How to Run This Project

Follow these steps in order to get the application running.

Step 1: Install Dependencies

You will need several Python libraries. You can install them all using pip:

pip install pandas scikit-learn imbalanced-learn matplotlib seaborn streamlit joblib


Step 2: Get the Data

This project requires the "Credit Card Fraud Detection" dataset.

1.Go to Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

2.Download the dataset.

3.Find the creditcard.csv file and save it in the same folder as the Python scripts.

Step 3: Train and Save the Model

Before you can run the app, you must first train the model and generate the .pkl files.

Run the following command in your terminal:

python fraud_detection_save_model.py


This script will load creditcard.csv, train the model (which may take a moment), and save two new files in your directory:

·fraud_model.pkl

·scaler.pkl

Step 4: Run the Streamlit App

Once the model files are saved, you can launch the front-end application.

Run the following command in your terminal:

streamlit run fraud_app.py


Your web browser will automatically open, and you will see the Fraud Detection App interface.

How the App Works

The model was trained on 30 features (V1-V28, Time, and Amount). In a real-world scenario, the V1-V28 features would be provided by the payment processor.

Since a user cannot enter these 28 values, this demo app:

·Asks the user for the Transaction Amount.

·Uses 0 as a default dummy value for all V1-V28 features.

·Uses the loaded scaler to preprocess the Amount.

·Feeds this data into the loaded model to get a prediction.