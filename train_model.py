import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import sys
import joblib  # <-- New import for saving the model

try:
    data = pd.read_csv("creditcard.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("--- FATAL ERROR ---")
    print("Error: 'creditcard.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same folder as this script.")
    print("Search for 'Credit Card Fraud Detection' on Kaggle.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit()

print("\n--- Data Exploration ---")
print("Data head:")
print(data.head())

print("\nData distribution (Fraud vs. Normal):")
class_distribution = data['Class'].value_counts()
print(class_distribution)
print(f"Percentage of Fraud: {class_distribution[1] / class_distribution.sum() * 100:.4f}%")

plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=data, palette=['#4376c0', '#d9534f'])
plt.title('Class Distribution (0: Normal, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Transaction Count')
plt.xticks([0, 1], ['Normal (0)', 'Fraud (1)'])
plt.show()

print("\n--- Data Preprocessing ---")

scaler = StandardScaler()
data['Scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time', 'Amount'], axis=1)

print("Data preprocessed: 'Amount' scaled and 'Time' dropped.")

X = data.drop('Class', axis=1)
y = data['Class']

print("\n--- Handling Imbalance with SMOTE ---")
print(f"Original shape of X: {X.shape}")
print(f"Original distribution:\n{y.value_counts()}")

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("\n--- After SMOTE ---")
print(f"Resampled shape of X: {X_res.shape}")
print(f"Resampled distribution:\n{y_res.value_counts()}")

print("\n--- Model Training ---")

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training the Random Forest model... (This may take a few moments)")
model.fit(X_train, y_train)
print("Model training complete.")

# --- NEW LINES ADDED ---
# After training, save the model and the scaler to disk
try:
    joblib.dump(model, 'fraud_model.pkl')
    # We must also save the scaler to process live transactions the exact same way
    joblib.dump(scaler, 'scaler.pkl') 
    print("Model and scaler saved to 'fraud_model.pkl' and 'scaler.pkl'")
except Exception as e:
    print(f"Error saving model: {e}")
# -----------------------

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

print("\n--- Process Finished ---")
