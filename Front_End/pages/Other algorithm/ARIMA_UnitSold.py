import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class UnitSoldARIMA:
    def __init__(self, store, product):
        self.store = store
        self.product = product

        # Load and preprocess data
        df = pd.read_csv('Data/Data_Set.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df = df.set_index('Date')
        df = df[(df['Store ID'] == store) & (df['Product ID'] == product)]

        df['Units Sold'] = df.groupby('Date')['Units Sold'].sum()
        df = self.feature_engineering(df)

        df = df.drop(columns=['Product ID', 'Store ID'])
        df.fillna(0, inplace=True)

        self.df = df
        self.category = df['Category'].iloc[0] if 'Category' in df.columns and not df.empty else 'Unknown'

    def feature_engineering(self, df):
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.day_of_week
        df['Quarter'] = df.index.quarter
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['DayOfYear'] = df.index.day_of_year
        df['Season'] = (df.index.month % 12) // 3 + 1

        df['month_sin'] = np.sin(2 * np.pi * df.Month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.Month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df.Day / 30)
        df['day_cos'] = np.cos(2 * np.pi * df.Day / 30)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.DayOfYear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.DayOfYear / 365)
        df['quarter_sin'] = np.sin(2 * np.pi * df.Quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df.Quarter / 4)
        df['season_sin'] = np.sin(2 * np.pi * df.Season / 4)
        df['season_cos'] = np.cos(2 * np.pi * df.Season / 4)

        # Lag Features
        for lag in [7, 14, 30]:
            df[f'Units_Sold_Lag_{lag}'] = df['Units Sold'].shift(lag)

        # Rolling Stats
        df['Rolling_Mean_7'] = df['Units Sold'].rolling(window=7).mean()
        df['Rolling_Std_7'] = df['Units Sold'].rolling(window=7).std()

        return df

    def train_model(self):
        data = self.df[['Units Sold']].dropna()

        # Splitting data into training and testing
        split_date = "2023-06-01"
        train = data.loc[data.index <= split_date].copy()
        test = data.loc[data.index > split_date].copy()

        # Fit ARIMA model on training data
        print("\nTraining ARIMA model...")
        self.model = ARIMA(train['Units Sold'], order=(2, 1, 2))  # You can adjust (p,d,q) here
        self.model_fit = self.model.fit()

        # Forecasting
        forecast = self.model_fit.forecast(steps=len(test))
        test['prediction'] = forecast.values

        # Evaluate and plot
        self.evaluate_model(test['Units Sold'], test['prediction'])
        self.plot_results(train, test)

    def evaluate_model(self, y_true, y_pred):
        print("\nModel Evaluation (ARIMA Model):")
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("MSE:", mean_squared_error(y_true, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
        print("R² Score:", r2_score(y_true, y_pred))

    def plot_results(self, train, test):
        plt.figure(figsize=(15, 6))
        plt.plot(train['Units Sold'], label='Training Data')
        plt.plot(test['Units Sold'], label='Test Data', color='green')
        plt.plot(test['prediction'], label='Prediction', color='red', linestyle='--')
        plt.legend()
        plt.title("Actual vs Predicted Sales (ARIMA)")
        plt.show()

    def predict_future(self, periods=30):
        # Forecast future periods
        print(f"\nForecasting next {periods} days...")
        future_forecast = self.model_fit.forecast(steps=periods)
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')

        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Units_Sold': future_forecast.values})
        future_df['StoreID'] = self.store
        future_df['ProductID'] = self.product
        future_df['Category'] = self.category

        self.plot_future(future_df)
        return future_df

    def plot_future(self, future_df):
        plt.figure(figsize=(15, 5))
        plt.plot(future_df['Date'], future_df['Predicted_Units_Sold'], linestyle='--', color='r')
        plt.title(f"Future Unit Sold Prediction for Store {self.store} - Product {self.product}")
        plt.xlabel("Date")
        plt.ylabel("Predicted Units Sold")
        plt.grid()
        plt.show()


# Example usage:
model = UnitSoldARIMA('S001', 'P0001')
model.train_model()
model.predict_future(30)  # Predict next 30 days
