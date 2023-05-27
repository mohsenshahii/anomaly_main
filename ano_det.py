import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load the real data from a CSV file
data = pd.read_csv('your_data.csv')

# Extract the features from the data (assuming all columns except the target variable are features)
features = data.drop('target_variable', axis=1)

# Split the data into training and testing sets
train_data, test_data = train_test_split(features, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Autoencoder model
input_dim = train_data.shape[1]
encoding_dim = 5

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
model.fit(train_data, train_data, epochs=50, batch_size=32, verbose=1)

# Perform reconstruction on the test data
reconstructed_data = model.predict(test_data)

# Calculate the reconstruction error (MSE) for each sample
mse = np.mean(np.power(test_data - reconstructed_data, 2), axis=1)

# Define a threshold for anomaly detection (e.g., using mean + 3 * standard deviation)
threshold = np.mean(mse) + 3 * np.std(mse)

# Identify anomalies
anomalies = test_data[mse > threshold]

# Plot the data
plt.scatter(test_data[:, 0], test_data[:, 1], color='blue', label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomaly')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection using Autoencoders')
plt.legend()
plt.show()