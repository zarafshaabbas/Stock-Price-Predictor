import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#  Download stock data
data = yf.download('AAPL', start='2018-01-01', end='2023-12-31')
data = data[['Close']]
data['Days'] = np.arange(len(data)).reshape(-1, 1)

#  Prepare features and labels
X = data['Days'].values.reshape(-1, 1)
y = data['Close'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Price')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
