# Sales & Demand Forecasting for Businesses
# Machine Learning Task 1 – Future Interns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Load Excel Dataset
# -----------------------------

data = pd.read_excel("Online Retail.xlsx")

# Limit dataset size for faster processing
data = data.head(50000)

print("\nDataset Preview:")
print(data.head())

# -----------------------------
# 2. Data Cleaning
# -----------------------------

# Remove rows with missing values
data = data.dropna()

# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Remove negative quantities (returns)
data = data[data['Quantity'] > 0]

# -----------------------------
# 3. Create Daily Sales Data
# -----------------------------

# Aggregate total quantity sold per day
daily_sales = data.groupby(data['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()

daily_sales.columns = ['date', 'sales']

# Convert date column back to datetime
daily_sales['date'] = pd.to_datetime(daily_sales['date'])

print("\nDaily Sales Preview:")
print(daily_sales.head())

# -----------------------------
# 4. Feature Engineering
# -----------------------------

daily_sales['year'] = daily_sales['date'].dt.year
daily_sales['month'] = daily_sales['date'].dt.month
daily_sales['day'] = daily_sales['date'].dt.day

# Numeric index for regression
daily_sales['day_number'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days

# -----------------------------
# 5. Visualize Historical Sales
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(daily_sales['date'], daily_sales['sales'], marker='o')

plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Items Sold")

plt.grid(True)

plt.show()

# -----------------------------
# 6. Train Forecasting Model
# -----------------------------

X = daily_sales[['day_number']]
y = daily_sales['sales']

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 7. Model Evaluation
# -----------------------------

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)

print("\nModel Evaluation")
print("Mean Absolute Error (MAE):", mae)

# -----------------------------
# 8. Forecast Future Demand
# -----------------------------

future_days = np.arange(
    daily_sales['day_number'].max() + 1,
    daily_sales['day_number'].max() + 31
)

future_predictions = model.predict(future_days.reshape(-1,1))

# -----------------------------
# 9. Save Forecast Results
# -----------------------------

forecast_df = pd.DataFrame({
    "Future_Day_Number": future_days,
    "Predicted_Sales": future_predictions
})

forecast_df.to_csv("forecast_output.csv", index=False)

print("\nFuture Sales Predictions:")
print(forecast_df.head())

# -----------------------------
# 10. Visualization: Actual vs Forecast
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(daily_sales['day_number'], daily_sales['sales'],
         label="Actual Sales", marker='o')

plt.plot(future_days, future_predictions,
         label="Forecasted Sales", linestyle="dashed")

plt.title("Sales Demand Forecast")
plt.xlabel("Day Number")
plt.ylabel("Sales Quantity")

plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# 11. Business Insight
# -----------------------------

print("\nBusiness Insight:")
print("This model forecasts future product demand using historical transaction data.")
print("Businesses can use this forecast to plan inventory levels,")
print("optimize staffing, and prepare for future demand changes.")