import os
import pandas as pd
import matplotlib.pyplot as plt
from timeseries.predictor import train_model, predict_next_days, load_data

# 🔁 1. Train model (if needed)
if not os.path.exists("models/lstm_model.pt"):
    train_model()

# 📊 2. Get data and generate forecast
df = load_data()
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
forecast = predict_next_days(n_days=7)

# 📅 3. Prepare plot
historical = df[-60:]  # last 60 days
last_date = historical['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=8, freq='D')[1:]

plt.figure(figsize=(10, 5))
plt.plot(historical['Date'], historical['Close'], label='History')
plt.plot(future_dates, forecast, label='Forecast', linestyle='--', marker='o')
plt.title('Price Forecast (Next 7 Days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()