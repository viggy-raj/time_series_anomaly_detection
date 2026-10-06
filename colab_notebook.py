# pip install ...


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shap
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


# Download MSL dataset from Telemanom
import urllib.request
import zipfile

# Telemanom data bundle
data_url = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"

os.makedirs('data', exist_ok=True)

print("Downloading MSL/SMAP dataset (this may take a few minutes)...")
urllib.request.urlretrieve(data_url, 'data/data.zip')

print("Extracting...")
with zipfile.ZipFile('data/data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

print("Dataset downloaded and extracted successfully!")
print("\nAvailable channels:")
print(os.listdir('data/data/MSL'))


# Load one MSL channel
channel_name = 'M-6'  # You can change this to any channel from the list above

# Load train and test data
x_train = np.load(f'data/data/MSL/{channel_name}_train.npy')
x_test = np.load(f'data/data/MSL/{channel_name}_test.npy')

# Load labels
import csv
with open(f'data/data/MSL/{channel_name}_test_labels.csv') as f:
    reader = csv.reader(f)
    labels_list = list(reader)

# Create binary labels
y_test = np.zeros(len(x_test))
for row in labels_list:
    start, end = int(row[0]), int(row[1])
    y_test[start:end] = 1

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Anomaly ratio: {y_test.mean():.4f}")

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create sequences
def create_sequences(data, labels, seq_length=50):
    sequences = []
    seq_labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        # Label is 1 if any point in sequence is anomalous
        seq_labels.append(1 if labels[i:i+seq_length].sum() > 0 else 0)
    return np.array(sequences), np.array(seq_labels)

SEQ_LENGTH = 50
X_train, _ = create_sequences(x_train_scaled, np.zeros(len(x_train_scaled)), SEQ_LENGTH)
X_test, y_test_seq = create_sequences(x_test_scaled, y_test, SEQ_LENGTH)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test_seq.shape}")
print(f"Anomaly ratio in sequences: {y_test_seq.mean():.4f}")


# Build LSTM Autoencoder
inputs = keras.Input(shape=(SEQ_LENGTH, X_train.shape[2]))
x = layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
x = layers.LSTM(32, activation='tanh', return_sequences=False)(x)
x = layers.RepeatVector(SEQ_LENGTH)(x)
x = layers.LSTM(32, activation='tanh', return_sequences=True)(x)
outputs = layers.LSTM(64, activation='tanh', return_sequences=True)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()


# Train
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.savefig('training_history.png')
plt.show()


# Predict and compute reconstruction error
X_test_pred = model.predict(X_test)
mse = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

# Threshold using percentile
threshold = np.percentile(mse, 95)
predictions = (mse > threshold).astype(int)

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test_seq, predictions, average='binary')
try:
    auc = roc_auc_score(y_test_seq, mse)
except:
    auc = 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUROC: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_seq, predictions)
print(f"Confusion Matrix:\n{cm}")


# SHAP explanation (simplified for time-series)
# Use a small sample for SHAP (computationally expensive)
sample_size = 50
X_sample = X_test[:sample_size]

# SHAP DeepExplainer
explainer = shap.DeepExplainer(model, X_sample)
shap_values = explainer.shap_values(X_sample[:5])

# Plot SHAP
shap.summary_plot(shap_values, X_sample[:5])
plt.savefig('shap_summary.png')
plt.show()


# Save metrics
results = {
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'auc': float(auc),
    'threshold': float(threshold),
    'confusion_matrix': cm.tolist()
}

import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results.json")
print("\nCopy the content of results.json and the plots to share for paper writing.")

