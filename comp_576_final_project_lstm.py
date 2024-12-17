"""

# Step 1: Process and label the data

###**ATTENTION: Can skip "Step 1" by the following steps!**


1.   Upload the processed and labeled data files – train.zip, validation.zip, test.zip to your Google drive.
2.   Unzip the files.
"""

from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/train.zip -d /content/data
!unzip /content/drive/MyDrive/validation.zip -d /content/data
!unzip /content/drive/MyDrive/test.zip -d /content/data

"""## 1.1 Upload the data
1.   Download data files from https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?resource=download.
2.   Place all the data from the 'Stocks' folders into a new folder named 'stocks'.
3.   Compress the 'stocks' folder.
4.   Upload 'stocks.zip' to '/content/data'

### NOTE

*   Remove a folder:
```
!rm -rf my_folder
```

## 1.2 Unzip the data
Unzip 'stock.zip' to the path '/content/data'
"""

!unzip /content/data/stocks.zip -d /content/data

"""## 1.3 Process and label the data
Overwrite all .txt files in 'stocks' folder:
1.   Delete the first line: 'Date,Open,High,Low,Close,Volume,OpenInt'
2.   Delete OpenInt column
3.   Calculate and add 5DayAvgVolume column
4.   Calculate and add Label column
---

### The final data format:
Date,Open,High,Low,Close,Volume,5DayAvgVolume,Label

### NOTE:
1. 5DayAvgVolume: int = the average volume of the previous 5 days rounded to the nearest integer
2. Label: int = whether the volume of the day is 3 times greater than the 5DayAvgVolume; if so, label = 1; o.w. label = 0
"""

import os
import pandas as pd
from glob import glob

def process_and_label_file(file_path):
    # Load the data, skipping the header row
    df = pd.read_csv(file_path, header=None, skiprows=1,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])

    # Calculate 5-day rolling average of volume, shifted to not include the current day
    df['5DayAvgVolume'] = df['Volume'].rolling(window=5).mean().shift().round().fillna(0).astype(int)

    # Label calculation
    df['Label'] = (df['Volume'] > 3 * df['5DayAvgVolume']).astype(int)

    # Drop the first 5 rows of actual data (after skipping the header) and the 'OpenInt' column
    df = df.iloc[5:].drop(columns=['OpenInt'])

    # Save the modified DataFrame back to the file
    df.to_csv(file_path, index=False, header=False)

def modify_files(directory_path: str):
    # Find all .txt files in the directory
    txt_files = glob(os.path.join(directory_path, '*.txt'))

    # Process each file
    for file_path in txt_files:
        process_and_label_file(file_path)

    print(f"Processed {len(txt_files)} files for {directory_path}.")

modify_files('/content/data/stocks')

"""## 1.4 Split the data into train/validation/test sets
1. Split the data randomly into 3 sets with the ratio **60:20:20** for **train:validation:test**.
2. Place the data to the path:
*   train data: '/content/data/train'
*   validation data: '/content/data/validation'
*   test data: '/content/data/test'
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Function to move files to the specified directory
def move_files(file_list, target_dir):
    for filename in file_list:
        shutil.move(os.path.join(source_dir, filename), os.path.join(target_dir, filename))

# Paths for the source and target directories
source_dir = '/content/data/stocks'
train_dir = '/content/data/train'
validation_dir = '/content/data/validation'
test_dir = '/content/data/test'

# Ensure target directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all .txt files in the source directory
all_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]

# Split the files for training and the rest (temporarily holding validation and test)
train_files, temp_files = train_test_split(all_files, test_size=0.4, random_state=42)

# Split the remaining files for validation and test
validation_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)


# Move the files to their respective directories
move_files(train_files, train_dir)
move_files(validation_files, validation_dir)
move_files(test_files, test_dir)

print(f"Moved {len(train_files)} files to {train_dir}, {len(validation_files)} to {validation_dir}, and {len(test_files)} to {test_dir}.")

"""---

# Step 2: Load data, adust labels, and select features
"""

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

"""#Step 3: Train LSTM(Long Short-Term Memory) Model
LSTM is a type of recurrent neural network (RNN) architecture designed to learn and capture dependencies over long sequences(i.e. handle long-term dependencies effectively).
LSTMs are widely used in financial time-series forecasting and have proven effective for capturing temporal dependencies in stock market data.

"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Convert data to NumPy arrays with float32 type
X_train_lstm = np.array(X_train.values, dtype=np.float32).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_validation_lstm = np.array(X_validation.values, dtype=np.float32).reshape((X_validation.shape[0], 1, X_validation.shape[1]))
X_test_lstm = np.array(X_test.values, dtype=np.float32).reshape((X_test.shape[0], 1, X_test.shape[1]))

# Convert target labels to NumPy arrays
y_train = np.array(y_train.values, dtype=np.float32)
y_validation = np.array(y_validation.values, dtype=np.float32)
y_test = np.array(y_test.values, dtype=np.float32)

# Define LSTM model using Input() for input shape definition
from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_validation_lstm, y_validation))

# Evaluate on validation and test data
y_validation_pred_proba = model.predict(X_validation_lstm).ravel()
y_test_pred_proba = model.predict(X_test_lstm).ravel()

y_validation_pred = (y_validation_pred_proba > 0.5).astype(int)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)

validation_accuracy = accuracy_score(y_validation, y_validation_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

validation_report = classification_report(y_validation, y_validation_pred)
test_report = classification_report(y_test, y_test_pred)

print("Validation Accuracy:", validation_accuracy)
print("Test Accuracy:", test_accuracy)
print("\nValidation Classification Report:\n", validation_report)
print("\nTest Classification Report:\n", test_report)

# Precision-Recall Curve for validation set
precision, recall, _ = precision_recall_curve(y_validation, y_validation_pred_proba)
auc_pr = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, marker='.', label=f'Validation PR Curve (AUC={auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation)')
plt.legend()
plt.show()

# Precision-Recall Curve for test set
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)
auc_pr_test = auc(recall_test, precision_test)

plt.figure()
plt.plot(recall_test, precision_test, marker='.', label=f'Test PR Curve (AUC={auc_pr_test:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test)')
plt.legend()
plt.show()
