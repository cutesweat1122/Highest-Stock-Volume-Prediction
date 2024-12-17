# -*- coding: utf-8 -*-
"""
# Step 1: Load the processed and labeled data files – train.zip, validation.zip from your Google Drive
"""

from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/train.zip -d /content/data
!unzip /content/drive/MyDrive/validation.zip -d /content/data
!unzip /content/drive/MyDrive/test.zip -d /content/data

"""# Step 2: Load data, adust labels, and select features"""

import pandas as pd
import os
import glob

# Function to load data from .txt files into a DataFrame
def load_data(directory):
    all_data = pd.DataFrame()
    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        current_data = pd.read_csv(file_path, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', '5DayAvgVolume', 'Label'])
        all_data = pd.concat([all_data, current_data], ignore_index=True)
    return all_data

directories = {
    'train': '/content/data/train',
    'validation': '/content/data/validation/',
    'test': '/content/data/test/'
}

# Load the data
train_data = load_data(directories['train'])
validation_data = load_data(directories['validation'])
test_data = load_data(directories['test'])

# Preprocess the data to create a binary target variable for peak volume days
train_data['Label'] = train_data['Label'].astype(int) == 1
validation_data['Label'] = validation_data['Label'].astype(int) == 1
test_data['Label'] = test_data['Label'].astype(int) == 1

# Select features and target variable
X_train = train_data[['Open', 'High', 'Low', 'Close', 'Volume', '5DayAvgVolume']]
y_train = train_data['Label']

X_validation = validation_data[['Open', 'High', 'Low', 'Close', 'Volume', '5DayAvgVolume']]
y_validation = validation_data['Label']

X_test = test_data[['Open', 'High', 'Low', 'Close', 'Volume', '5DayAvgVolume']]
y_test = test_data['Label']

"""# Step 3: Train TFT(Temporal Fusion Transformer) Model
TFT excels in time-series forecasting tasks by leveraging self-attention mechanisms to capture both short and long-term dependencies.
"""

import tensorflow as tf
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Define the TFT model using the Keras Functional API
def build_tft_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Feature processing: Dense layers for demonstration
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Build the model
input_shape = X_train.shape[1:]  # Number of features
tft_model = build_tft_model(input_shape)

# Compile the model
tft_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
tft_model.fit(
    X_train, y_train,
    validation_data=(X_validation, y_validation),
    epochs=20,
    batch_size=32
)

# Predict on validation and test sets
y_validation_pred_probs = tft_model.predict(X_validation).flatten()
y_test_pred_probs = tft_model.predict(X_test).flatten()

# Convert probabilities to binary predictions (threshold = 0.5)
y_validation_pred = (y_validation_pred_probs >= 0.5).astype(int)
y_test_pred = (y_test_pred_probs >= 0.5).astype(int)

# Evaluate the model
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
validation_report = classification_report(y_validation, y_validation_pred)
test_report = classification_report(y_test, y_test_pred)

print("Validation Accuracy:", validation_accuracy)
print("Test Accuracy:", test_accuracy)
print("\nValidation Classification Report:\n", validation_report)
print("\nTest Classification Report:\n", test_report)

# Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred_probs, title):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Generate Precision-Recall Curve for validation and test sets
plot_precision_recall_curve(y_validation, y_validation_pred_probs, "Validation Set Precision-Recall Curve")
plot_precision_recall_curve(y_test, y_test_pred_probs, "Test Set Precision-Recall Curve")
