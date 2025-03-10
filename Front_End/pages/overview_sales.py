import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

class StoreSalesPredictor:
    def __init__(self, data_path):
        """Initialize with dataset path, load data, and preprocess"""
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'], dayfirst=True)  # Fix date format
        self.models = {}  # Store trained models per store

    def preprocess_data(self):
        """Prepares the dataset for training"""
        self.df = self.df[['Date', 'Store ID', 'Product ID', 'Units Sold']]
        self.df.sort_values(['Store ID', 'Date'], inplace=True)

    def train_model(self, store_id):
        """Train XGBoost model for a specific Store ID"""
        store_data = self.df[self.df['Store ID'] == store_id].copy()

        # Aggregate total units sold across all 20 products per day
        daily_sales = store_data.groupby('Date')['Units Sold'].sum().reset_index()

        # Feature Engineering
        daily_sales['Day'] = daily_sales['Date'].dt.day
        daily_sales['Month'] = daily_sales['Date'].dt.month
        daily_sales['Year'] = daily_sales['Date'].dt.year
        daily_sales['Lag_1'] = daily_sales['Units Sold'].shift(1)  # Previous day's sales

        # Drop NaN values
        daily_sales.dropna(inplace=True)

        # Define Features and Target
        X = daily_sales[['Day', 'Month', 'Year', 'Lag_1']]
        y = daily_sales['Units Sold']

        # Train-Test Split
        X_train, y_train = X[:-1], y[:-1]  # Train on all except last entry

        # Train XGBoost Model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)

        self.models[store_id] = model

    def predict_store(self, store_id):
        """Predict today's total Units Sold for a specific Store ID (sum of all Product IDs)"""
        today = self.df['Date'].max() + timedelta(days=1)  # Predict for tomorrow
        store_data = self.df[self.df['Store ID'] == store_id].copy()

        if store_data.empty:
            return {store_id: "No data available"}

        # Aggregate total units sold across all 20 products for the last recorded date
        last_date = store_data['Date'].max()
        last_day_sales = store_data[store_data['Date'] == last_date]['Units Sold'].sum()

        # Prepare input features for prediction
        X_test = pd.DataFrame({
            'Day': [today.day],
            'Month': [today.month],
            'Year': [today.year],
            'Lag_1': [last_day_sales]  # Use aggregated sum as previous day's sales
        })

        # Make prediction
        model = self.models.get(store_id)
        if model:
            y_pred = model.predict(X_test)
            return {store_id: round(y_pred[0])}
        else:
            return {store_id: "Model not trained"}

    def train_all(self):
        """Train models for all stores"""
        self.preprocess_data()
        for store_id in self.df['Store ID'].unique():
            self.train_model(store_id)

# Usage
if __name__ == "__main__":
    predictor = StoreSalesPredictor("Data/Data_Set.csv")
    predictor.train_all()  

    print(predictor.predict_store("S001"))
    print(predictor.predict_store("S002"))
    print(predictor.predict_store("S003"))
    print(predictor.predict_store("S004"))
    print(predictor.predict_store("S005"))
    
