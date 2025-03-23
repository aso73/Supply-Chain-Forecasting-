import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

class UnitSold:
    def __init__(self, store, product):
        self.store = store
        self.product = product

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
        df['Season'] = (df.index.month) % 12 // 3 + 1

        df['month_sin'] = np.sin(2*np.pi*df.Month/12)  
        df['month_cos'] = np.cos(2*np.pi*df.Month/12)
        df['day_sin'] = np.sin(2*np.pi*df.Day/30)
        df['day_cos'] = np.cos(2*np.pi*df.Day/30)
        df['day_of_year_sin'] = np.sin(2*np.pi*df.DayOfYear/365)
        df['day_of_year_cos'] = np.cos(2*np.pi*df.DayOfYear/365)
        df['quarter_sin'] = np.sin(2*np.pi*df.Quarter/4)
        df['quarter_cos'] = np.cos(2*np.pi*df.Quarter/4)
        df['season_sin'] = np.sin(2*np.pi*df.Season/4)
        df['season_cos'] = np.cos(2*np.pi*df.Season/4)

        for lag in [7, 14, 30]:
            df[f'Units_Sold_Lag_{lag}'] = df['Units Sold'].shift(lag)
        
        df['Rolling_Mean_7'] = df['Units Sold'].rolling(window=7).mean()
        df['Rolling_Std_7'] = df['Units Sold'].rolling(window=7).std()

        return df
    
    def train_model(self):
        data = self.df.copy()
        data.fillna(0, inplace=True)  # Fill missing values for SVR

        split_date = "2023-06-01"
        train = data.loc[data.index <= split_date].copy()
        test = data.loc[data.index > split_date].copy()

        FEATURES = [col for col in data.columns if col not in ['Units Sold', 'prediction', 'Category']]
        TARGET = 'Units Sold'
        
        X_train, Y_train = train[FEATURES], train[TARGET]
        X_test, Y_test = test[FEATURES], test[TARGET]

        # Scaling
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train.values.reshape(-1, 1)).ravel()

        # SVR model
        self.model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
        self.model.fit(X_train_scaled, Y_train_scaled)

        predictions_scaled = self.model.predict(X_test_scaled)
        predictions = self.scaler_Y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

        test['prediction'] = predictions
        data = data.merge(test[['prediction']], how='left', left_index=True, right_index=True)

        self.evaluate_model(Y_test, predictions)
        self.plot_results(data)
    
    def evaluate_model(self, Y_test, predictions):
        print("Model Evaluation (SVR Model)")
        print("MAE:", mean_absolute_error(Y_test, predictions))
        print("MSE:", mean_squared_error(Y_test, predictions))
        print("RMSE:", np.sqrt(mean_squared_error(Y_test, predictions)))
        print("R² Score:", r2_score(Y_test, predictions))

    def plot_results(self, data):
        ax = data[['Units Sold']].plot(figsize=(15, 5))
        data['prediction'].plot(ax=ax, style='--')
        plt.legend(['True Data', 'Prediction'])
        plt.title("Actual vs Predicted Sales using SVR")
        plt.show()

    def predict_future(self, start_date, end_date, past_year=2023):
        future_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        future_df = pd.DataFrame(index=future_dates)
        
        past_start = f"{past_year}-{start_date[5:]}"  
        past_end = f"{past_year}-{end_date[5:]}"
        past_df = self.df.loc[past_start:past_end]
        
        if past_df.empty:
            raise ValueError(f"No historical data found for {past_year} in the given date range.")
        
        random_multiplier = np.random.uniform(0.4, 0.9, size=len(future_dates))
        
        if len(past_df) < len(future_dates):
            past_values = np.resize(past_df["Units Sold"].values, len(future_dates))
        else:
            past_values = past_df["Units Sold"].values[:len(future_dates)]
        
        future_df["Predicted_Units_Sold"] = past_values * random_multiplier
        future_df["Predicted_Units_Sold"].fillna(future_df["Predicted_Units_Sold"].mean(), inplace=True)
        
        future_df.reset_index(inplace=True)
        future_df.rename(columns={'index': 'Date'}, inplace=True)
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


# Usage Example
model = UnitSold('S001', 'P0001')
model.train_model()
model.predict_future("2026-06-01", "2026-06-30")
