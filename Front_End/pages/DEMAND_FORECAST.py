import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

class DemandForecastingModel:
    def __init__(self):
        self.filepath = "Data/Data_Set.csv"
        self.df = self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        df = pd.read_csv(self.filepath)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.set_index('Date')
        
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.day_of_week
        df['Quater'] = df.index.quarter
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['DayOfYear'] = df.index.dayofyear
        df['Season'] = (df.index.month) % 12 // 3 + 1
        
        days_in_month = 30
        df['month_sin'] = np.sin(2 * np.pi * df.Month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.Month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df.Day / days_in_month)
        df['day_cos'] = np.cos(2 * np.pi * df.Day / days_in_month)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.DayOfYear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.DayOfYear / 365)
        df['quarter_sin'] = np.sin(2 * np.pi * df.Quater / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df.Quater / 4)
        df['season_sin'] = np.sin(2 * np.pi * df.Season / 4)
        df['season_cos'] = np.cos(2 * np.pi * df.Season / 4)
        
        return df
    
    def predict_demand_product(self, product_id, year=None):
        data = self.df[self.df['Product ID'] == product_id]
        
        if year is None:
            split_date = "2023-01-01"
            train = data.loc[data.index <= split_date].copy()
            test = data.loc[data.index > split_date].copy()
            
            FEATURES = ['Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Day',
                        'DayOfWeek', 'Quater', 'Month', 'Year', 'DayOfYear', 'Season', 'month_sin', 'month_cos',
                        'day_sin', 'day_cos', 'day_of_year_sin', 'day_of_year_cos', 'quarter_sin', 'quarter_cos',
                        'season_sin', 'season_cos']
            TARGET = 'Demand Forecast'
            
            X_train = train[FEATURES]
            Y_train = train[TARGET]
            X_test = test[FEATURES]
            Y_test = test[TARGET]
            
            reg = xgb.XGBRegressor(n_estimators=1000)
            reg.fit(X_train, Y_train,
                    eval_set=[(X_train, Y_train), (X_test, Y_test)],
                    verbose=100
                    )
            
            test['prediction'] = reg.predict(X_test)
            data = data.merge(test[['prediction']],
                              how='left',
                              left_index=True,
                              right_index=True)
            
            data['Month'] = data.index.month
            monthly_predictions = data.groupby('Month')['prediction'].mean()
            
            monthly_predictions.plot(kind='bar', figsize=(15, 5), color='blue')
            plt.title(f"Average Monthly Demand Forecast for Product {product_id}")
            plt.xlabel("Month")
            plt.ylabel("Average Demand Forecast")
            plt.xticks(ticks=range(12), labels=[calendar.month_name[i] for i in range(1, 13)])
            # plt.show()
        else:
            future_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
            future_data = pd.DataFrame(index=future_dates)
            
            future_data['Day'] = future_data.index.day
            future_data['DayOfWeek'] = future_data.index.day_of_week
            future_data['Quater'] = future_data.index.quarter
            future_data['Month'] = future_data.index.month
            future_data['Year'] = future_data.index.year
            future_data['DayOfYear'] = future_data.index.dayofyear
            future_data['Season'] = (future_data.index.month) % 12 // 3 + 1
            future_data['month_sin'] = np.sin(2 * np.pi * future_data.Month / 12)
            future_data['month_cos'] = np.cos(2 * np.pi * future_data.Month / 12)
            future_data['day_sin'] = np.sin(2 * np.pi * future_data.Day / 30)
            future_data['day_cos'] = np.cos(2 * np.pi * future_data.Day / 30)
            future_data['day_of_year_sin'] = np.sin(2 * np.pi * future_data.DayOfYear / 365)
            future_data['day_of_year_cos'] = np.cos(2 * np.pi * future_data.DayOfYear / 365)
            future_data['quarter_sin'] = np.sin(2 * np.pi * future_data.Quater / 4)
            future_data['quarter_cos'] = np.cos(2 * np.pi * future_data.Quater / 4)
            future_data['season_sin'] = np.sin(2 * np.pi * future_data.Season / 4)
            future_data['season_cos'] = np.cos(2 * np.pi * future_data.Season / 4)
            
            FEATURES = ['Day', 'DayOfWeek', 'Quater', 'Month', 'Year', 'DayOfYear', 'Season', 
                        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_year_sin', 
                        'day_of_year_cos', 'quarter_sin', 'quarter_cos', 'season_sin', 'season_cos']
            
            X_train = data[FEATURES]
            Y_train = data['Demand Forecast']
            
            reg = xgb.XGBRegressor(n_estimators=1000)
            reg.fit(X_train, Y_train,
                    verbose=100)
            
            future_data['prediction'] = reg.predict(future_data[FEATURES])
            future_data['Product ID'] = product_id
            
            avg_predictions = future_data.groupby('Month')['prediction'].mean()
            
            avg_predictions.plot(kind='bar', figsize=(15, 5), color='purple')
            plt.title(f"Average Monthly Demand Forecast for Product {product_id} in {year}")
            plt.xlabel("Month")
            plt.ylabel("Average Demand Forecast")
            plt.xticks(ticks=range(12), labels=[calendar.month_name[i] for i in range(1, 13)])
            # plt.show()
    
    def predict_demand_month(self, month, year=None):
        if year is None:
            data = self.df[self.df['Month'] == month]
            FEATURES = ['Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Day',
                        'DayOfWeek', 'Quater', 'Month', 'Year', 'DayOfYear', 'Season', 'month_sin', 'month_cos',
                        'day_sin', 'day_cos', 'day_of_year_sin', 'day_of_year_cos', 'quarter_sin', 'quarter_cos',
                        'season_sin', 'season_cos']
            TARGET = 'Demand Forecast'
            
            predictions = []
            for product_id in data['Product ID'].unique():
                product_data = data[data['Product ID'] == product_id]
                X = product_data[FEATURES]
                Y = product_data[TARGET]
                
                reg = xgb.XGBRegressor(n_estimators=1000)
                reg.fit(X, Y,
                        eval_set=[(X, Y), (X, Y)],
                        verbose=100)
                
                product_data['prediction'] = reg.predict(X)
                predictions.append(product_data[['Product ID', 'prediction']])
            
            predictions_df = pd.concat(predictions)
            avg_predictions = predictions_df.groupby('Product ID')['prediction'].mean()
            
            avg_predictions.plot(kind='bar', figsize=(15, 5), color='green')
            month_name = calendar.month_name[month]
            plt.title(f"Average Demand Forecast for All Products in {month_name}")
            plt.xlabel("Product ID")
            plt.ylabel("Average Demand Forecast")
            # plt.show()
        else:
            future_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
            future_data = pd.DataFrame(index=future_dates)
            
            future_data['Day'] = future_data.index.day
            future_data['DayOfWeek'] = future_data.index.day_of_week
            future_data['Quater'] = future_data.index.quarter
            future_data['Month'] = future_data.index.month
            future_data['Year'] = future_data.index.year
            future_data['DayOfYear'] = future_data.index.dayofyear
            future_data['Season'] = (future_data.index.month) % 12 // 3 + 1
            future_data['month_sin'] = np.sin(2 * np.pi * future_data.Month / 12)
            future_data['month_cos'] = np.cos(2 * np.pi * future_data.Month / 12)
            future_data['day_sin'] = np.sin(2 * np.pi * future_data.Day / 30)
            future_data['day_cos'] = np.cos(2 * np.pi * future_data.Day / 30)
            future_data['day_of_year_sin'] = np.sin(2 * np.pi * future_data.DayOfYear / 365)
            future_data['day_of_year_cos'] = np.cos(2 * np.pi * future_data.DayOfYear / 365)
            future_data['quarter_sin'] = np.sin(2 * np.pi * future_data.Quater / 4)
            future_data['quarter_cos'] = np.cos(2 * np.pi * future_data.Quater / 4)
            future_data['season_sin'] = np.sin(2 * np.pi * future_data.Season / 4)
            future_data['season_cos'] = np.cos(2 * np.pi * future_data.Season / 4)
            
            FEATURES = ['Day', 'DayOfWeek', 'Quater', 'Month', 'Year', 'DayOfYear', 'Season', 
                        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_year_sin', 
                        'day_of_year_cos', 'quarter_sin', 'quarter_cos', 'season_sin', 'season_cos']
            
            predictions = []
            for product_id in self.df['Product ID'].unique():
                product_data = self.df[self.df['Product ID'] == product_id]
                X_train = product_data[FEATURES]
                Y_train = product_data['Demand Forecast']
                
                reg = xgb.XGBRegressor(n_estimators=1000)
                reg.fit(X_train, Y_train,
                    verbose=100)
                
                future_data['prediction'] = reg.predict(future_data[FEATURES])
                future_data['Product ID'] = product_id
                predictions.append(future_data[['Product ID', 'prediction']])
            
            predictions_df = pd.concat(predictions)
            avg_predictions = predictions_df.groupby('Product ID')['prediction'].mean()
            
            avg_predictions.plot(kind='bar', figsize=(15, 5), color='purple')
            month_name = calendar.month_name[month]
            plt.title(f"Average Demand Forecast for All Products in {month_name} {year}")
            plt.xlabel("Product ID")
            plt.ylabel("Average Demand Forecast")
            # plt.show()

# Usage Example:
# model = DemandForecastingModel()
# model.predict_demand_product("P0002", 2026) 
# model.predict_demand_month(3,  2026)