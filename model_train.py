import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import joblib

# Step 1: Load and prepare the data
df = pd.read_csv('Data\\weatherHistory.csv')
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])
df = df.sort_values('Formatted Date')

# Step 2: Select relevant features
features = ['Temperature (C)', 'Humidity', 'Pressure (millibars)']
data = df[features].values

# Step 3: Normalize using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Step 4: Create sequences for LSTM
def create_sequences(dataset, time_steps=24):
    X, y = [], []
    for i in range(len(dataset) - time_steps - 1):
        X.append(dataset[i:(i + time_steps)])
        y.append(dataset[i + time_steps])  # predict all 3 features
    return np.array(X), np.array(y)

time_steps = 24
X, y = create_sequences(data_scaled, time_steps)

# Step 5: Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 6: Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(time_steps, X.shape[2])))
model.add(Dense(3))  # 3 outputs: temperature, humidity, pressure
model.compile(optimizer='adam', loss=MeanSquaredError())  # ✅ Avoid "mse" string issue

# Step 7: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Step 8: Predict on test data
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Step 9: Inverse transform predictions
train_pred_inv = scaler.inverse_transform(train_pred)
test_pred_inv = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Step 10: Plot predictions for all 3 features
plt.figure(figsize=(14, 6))
plt.subplot(3, 1, 1)
plt.plot(y_test_inv[:500, 0], label='Actual Temp')
plt.plot(test_pred_inv[:500, 0], label='Predicted Temp')
plt.title("Temperature Prediction")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(y_test_inv[:500, 1], label='Actual Humidity')
plt.plot(test_pred_inv[:500, 1], label='Predicted Humidity')
plt.title("Humidity Prediction")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(y_test_inv[:500, 2], label='Actual Pressure')
plt.plot(test_pred_inv[:500, 2], label='Predicted Pressure')
plt.title("Pressure Prediction")
plt.legend()

plt.tight_layout()
plt.show()

# Step 11: Save model and scaler
model.save('weather_multifeature_lstm_model.h5')
joblib.dump(scaler, 'multifeature_scaler.save')
