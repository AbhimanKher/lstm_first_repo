
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from data_processing import load_data, preprocess_data, split_data, create_dataset
from model import build_model
import tensorflow as tf


# Load and preprocess data
df = load_data()
data, scaler = preprocess_data(df)
train_data, test_data = split_data(data)

# Create datasets
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train model
model = build_model(time_step)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Predict and inverse transform
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
print("Train RMSE:", np.sqrt(mean_squared_error(y_train_inv, train_predict)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test_inv, test_predict)))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_train)), y_train_inv, label='Train Actual')
plt.plot(np.arange(len(y_train)), train_predict, label='Train Predicted')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv, label='Test Actual')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_predict, label='Test Predicted')
plt.legend()
plt.title("Stock Price Prediction Using Stacked LSTM")
plt.show()
