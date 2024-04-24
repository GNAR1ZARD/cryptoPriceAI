############################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Name of the cryptocurrency from /past folder
crypto_name = 'bitcoin'

############################################################################################
# Training Model on past data
############################################################################################

# Load the past data
past = pd.read_csv(f'data/{crypto_name}.csv')

# Pre-processing
past = past.dropna()

# Feature Engineering: Extract day of week and month from date
past['date'] = pd.to_datetime(past['date'])
past['day_of_week'] = past['date'].dt.dayofweek
past['month'] = past['date'].dt.month

# Normalize features
scaler = StandardScaler()
# features = ['day_of_week', 'month']
features = ['total_volume', 'market_cap', 'day_of_week', 'month']
past[features] = scaler.fit_transform(past[features])

# Split the pastset
X = past[features]
y = past['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', root_mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

############################################################################################
# Showcasing model predictions on future data
############################################################################################

# Generate future dates (2000 days)
future_dates = [datetime.now() + timedelta(days=x) for x in range(1, 2000)]
future = pd.DataFrame(future_dates, columns=['date', 'volume', 'cap'])

# Convert 'date' to datetime and extract 'day_of_week' and 'month'
future['date'] = pd.to_datetime(future['date'])
future['day_of_week'] = future['date'].dt.dayofweek
future['month'] = future['date'].dt.month

# Normalize features
features_to_scale = ['total_volume', 'market_cap', 'day_of_week', 'month']
future[features_to_scale] = scaler.transform(future[features_to_scale])

# Predict future prices
future_prices = model.predict(future[features_to_scale])

# Create a plot with the original pastset and the predictions for the test set
plt.figure(figsize=(14, 7))

# Plotting the actual prices from past data
plt.plot(past['date'], y, label='Actual Prices', color='blue')

# aligning the test set predictions with the test set dates
test_dates = X_test.merge(past[['date']], how='left', left_index=True, right_index=True)['date']

# sort data
test_dates, y_pred_aligned = zip(*sorted(zip(test_dates, y_pred)))

# Plotting the predicted prices for the test set
plt.scatter(test_dates, y_pred_aligned, label='Predicted Prices (Test Set)', color='yellow', linestyle='--')

# Plotting the future predictions
plt.plot(future['date'], future_prices, label='Future Predicted Prices', color='red', linestyle='--')

# Adding labels and title
plt.title(f'{crypto_name.capitalize()} Prices: Actual, Predicted, and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

############################################################################################