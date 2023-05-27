import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Generate normal and anomalous data
np.random.seed(42)
normal_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10))
np.random.seed(42)
anomalous_data = np.random.normal(loc=5.0, scale=2.0, size=(100, 10))

# Combine normal and anomalous data
data = np.vstack((normal_data, anomalous_data))

# Shuffle the data
np.random.shuffle(data)

# Split the data into training and testing sets
train_data = data[:900]
test_data = data[900:]

# Normalize the data
scaler = preprocessing.StandardScaler()
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

# Calculate the mean squared error (MSE) for each sample
mse = np.mean(np.power(test_data - reconstructed_data, 2), axis=1)

# Define a threshold for anomaly detection
threshold = np.mean(mse) + np.std(mse)

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