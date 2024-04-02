# Project Proposal for CS 422 – Cryptocurrency Price Prediction Using Machine Learning

## Team Members
- Ryan Amendala
- Kamil Dusejovsky
- Cody Stumbough


## Problem Definition
Our project aims to tackle a regression problem within the domain of financial technology, specifically predicting the future prices of cryptocurrencies based on historical data. The volatility and unpredictable nature of cryptocurrency markets present a significant computational challenge, making accurate predictions valuable for investors and analysts alike.

## Machine Learning Methods
- **Linear Regression:** To establish a baseline for prediction accuracy.
- **LSTM (Long Short-Term Memory) Networks:** To leverage the temporal nature of the dataset, ideal for time series forecasting.

## Data Description
The dataset from Kaggle, "Cryptocurrency Historical Prices from CoinGecko," comprises historical price data of various cryptocurrencies. Key features include:
- Date and time of observation
- Cryptocurrency name and symbol
- Opening, closing, high, and low prices
- Volume and market capitalization

The dataset contains over 1 million records, capturing daily price movements across multiple cryptocurrencies.

## Data Settings and Preprocessing
- **Cleaning:** Removal of null or missing values, and correction of any anomalies in price data.
- **Normalization:** Scaling numerical features to a similar range to improve model performance.
- **Feature Engineering:** Extraction of time-based features (e.g., day of the week, month) to capture seasonal trends.
- **Data Partitioning:** Splitting the dataset into training, validation, and testing sets to evaluate model performance.

## Experiment Design
- **Model Training:** Each model will be trained on the historical price data, using 80% of the dataset for training and 20% for testing.
- **Parameter Tuning:** Utilizing grid search with cross-validation to find the optimal parameters for each model.
- **Evaluation Metrics:** Model performance will be evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) to quantify prediction accuracy.
- **Comparative Analysis:** Although a detailed performance comparison with benchmark methods is not required, we will present an analysis comparing the accuracy and efficiency of the chosen models.

## Timeline and Milestones
- **Week 1:** Data collection, cleaning, and preprocessing.
- **Week 2:** Model implementation and initial training.
- **Week 3:** Parameter tuning and model optimization.
- **Week 4:** Evaluation and analysis of model performance. Final report preparation and presentation.

## Conclusion and Future Work
In this project, we aim to apply and evaluate various machine learning techniques for predicting cryptocurrency prices, a task with substantial real-world applicability and computational challenges. Through this exploration, we hope to gain insights into the predictive capabilities of different models in the volatile cryptocurrency market. Future work could involve exploring more complex models, incorporating additional predictors such as news sentiment analysis, or expanding the scope to include more cryptocurrencies.
