#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized Sticker Sales Prediction Script
Performance improvements focused on eliminating bottlenecks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import holidays
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configure plotting
sns.set_style('whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

class OptimizedStickerSalesPredictor:
    """Optimized sticker sales prediction pipeline"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.models = {}
        self.holiday_cache = self._initialize_holiday_cache()
        
    def _initialize_holiday_cache(self):
        """Initialize holiday cache once to avoid repeated object creation"""
        return {
            'Canada': holidays.country_holidays('CA'),
            'Finland': holidays.country_holidays('FI'),
            'Italy': holidays.country_holidays('IT'),
            'Kenya': holidays.country_holidays('KE'),
            'Norway': holidays.country_holidays('NO'),
            'Singapore': holidays.country_holidays('SG')
        }
    
    def load_data(self, train_path: str, test_path: str) -> tuple:
        """Load and return train/test data efficiently"""
        print("Loading data...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Convert to datetime efficiently
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])
        
        return train_data, test_data
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values efficiently using vectorized operations"""
        df = df.copy()
        if 'num_sold' in df.columns:
            # Vectorized fillna using transform
            df['num_sold'] = df.groupby('country')['num_sold'].transform(
                lambda x: x.fillna(x.mean())
            )
        return df
    
    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date-based features efficiently"""
        df = df.copy()
        
        # Vectorized date feature extraction
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        return df
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """OPTIMIZED: Add holiday features using vectorized operations"""
        df = df.copy()
        df["holiday"] = 0
        
        # Vectorized approach using boolean indexing - MAJOR OPTIMIZATION
        for country, holiday_obj in self.holiday_cache.items():
            country_mask = df['country'] == country
            if country_mask.any():  # Only process if country exists
                # Get dates for this country and check against holidays
                country_dates = df.loc[country_mask, 'date']
                holiday_mask = country_dates.isin(holiday_obj.keys())
                # Set holiday flag using boolean indexing
                df.loc[country_mask & holiday_mask, 'holiday'] = 1
        
        return df
    
    def apply_one_hot_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Apply one-hot encoding efficiently"""
        categorical_cols = ['country', 'store', 'product']
        
        # Use pd.get_dummies which is optimized
        train_encoded = pd.get_dummies(train_df, columns=categorical_cols)
        test_encoded = pd.get_dummies(test_df, columns=categorical_cols)
        
        # Ensure both dataframes have same columns
        all_columns = set(train_encoded.columns) | set(test_encoded.columns)
        for col in all_columns:
            if col not in train_encoded.columns:
                train_encoded[col] = 0
            if col not in test_encoded.columns:
                test_encoded[col] = 0
        
        # Reorder columns to match
        train_encoded = train_encoded.reindex(columns=sorted(all_columns))
        test_encoded = test_encoded.reindex(columns=sorted(all_columns))
        
        return train_encoded, test_encoded
    
    def apply_periodic_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """OPTIMIZED: Apply sine/cosine transformation efficiently"""
        df = df.copy()
        cyclic_cols = ['month', 'day', 'day_of_week']
        
        # Vectorized periodic transformation - OPTIMIZATION
        for col in cyclic_cols:
            if col in df.columns:
                max_val = df[col].max()
                df[f"{col}_SIN"] = np.sin(df[col] / max_val * 2 * np.pi)
                df[f"{col}_COS"] = np.cos(df[col] / max_val * 2 * np.pi)
        
        return df
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Complete preprocessing pipeline with optimizations"""
        print("Preprocessing data...")
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        
        # Add date features
        train_df = self.add_date_features(train_df)
        test_df = self.add_date_features(test_df)
        
        # Add holiday features (MAJOR OPTIMIZATION HERE)
        train_df = self.add_holiday_features(train_df)
        test_df = self.add_holiday_features(test_df)
        
        # One-hot encoding
        train_encoded, test_encoded = self.apply_one_hot_encoding(train_df, test_df)
        
        # Periodic transformation (OPTIMIZATION HERE)
        train_encoded = self.apply_periodic_transform(train_encoded)
        test_encoded = self.apply_periodic_transform(test_encoded)
        
        # Drop unwanted columns
        drop_cols = ['month', 'day', 'day_of_week', 'date']
        if 'id' in train_encoded.columns:
            drop_cols.append('id')
        
        train_final = train_encoded.drop(columns=[col for col in drop_cols if col in train_encoded.columns])
        test_final = test_encoded.drop(columns=[col for col in drop_cols if col in test_encoded.columns])
        
        return train_final, test_final
    
    def split_features_target(self, df: pd.DataFrame) -> tuple:
        """Split features and target variable"""
        X = df.drop(['num_sold'], axis=1)
        y = df['num_sold']
        return X, y
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Train multiple models and return performance metrics"""
        print("Training models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_config = {
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),  # Use all cores
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
            
            print(f"{name} - MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
        
        # XGBoost
        print("Training XGBoost...")
        xgb_train = xgb.DMatrix(X_train_scaled, label=y_train)
        xgb_test = xgb.DMatrix(X_test_scaled, label=y_test)
        
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
            'n_estimators': 100,
            'random_state': 42
        }
        
        xgb_model = xgb.train(params, xgb_train, num_boost_round=100)
        xgb_pred = xgb_model.predict(xgb_test)
        
        # XGBoost metrics
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'mae': xgb_mae,
            'mse': xgb_mse,
            'rmse': xgb_rmse,
            'mape': xgb_mape,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled
        }
        
        print(f"XGBoost - MAPE: {xgb_mape:.4f}, RMSE: {xgb_rmse:.4f}")
        
        return results
    
    def get_best_model(self, results: dict) -> tuple:
        """Get the best model based on MAPE"""
        best_model_name = min(results.keys(), key=lambda x: results[x]['mape'])
        return best_model_name, results[best_model_name]
    
    def predict_test_data(self, test_final: pd.DataFrame, best_model_info: dict, 
                         test_ids: pd.Series) -> pd.DataFrame:
        """Make predictions on test data and create submission"""
        test_scaled = self.scaler.transform(test_final)
        
        if 'XGBoost' in str(type(best_model_info['model'])):
            test_dmatrix = xgb.DMatrix(test_scaled)
            predictions = best_model_info['model'].predict(test_dmatrix)
        else:
            predictions = best_model_info['model'].predict(test_scaled)
        
        submission_df = pd.DataFrame({
            'id': test_ids,
            'num_sold': predictions
        })
        
        return submission_df
    
    def run_pipeline(self, train_path: str, test_path: str) -> pd.DataFrame:
        """Run the complete optimized pipeline"""
        # Load data
        train_data, test_data = self.load_data(train_path, test_path)
        
        # Store test IDs
        test_ids = test_data['id'].copy()
        
        # Preprocess
        train_final, test_final = self.preprocess_data(train_data, test_data)
        
        # Split features and target
        X, y = self.split_features_target(train_final)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Train models
        results = self.train_models(X_train, y_train, X_test, y_test)
        
        # Get best model
        best_model_name, best_model_info = self.get_best_model(results)
        print(f"\nBest model: {best_model_name} with MAPE: {best_model_info['mape']:.4f}")
        
        # Make predictions on test data
        submission = self.predict_test_data(test_final, best_model_info, test_ids)
        
        return submission

def main():
    """Main function demonstrating optimized pipeline"""
    print("Starting Optimized Sticker Sales Prediction Pipeline...")
    
    # Note: Update these paths as needed
    train_path = "train.csv"  # Update with actual path
    test_path = "test.csv"    # Update with actual path
    
    try:
        predictor = OptimizedStickerSalesPredictor()
        submission = predictor.run_pipeline(train_path, test_path)
        
        # Save submission
        submission.to_csv('Optimized_Submission.csv', index=False)
        print("Optimized pipeline completed successfully!")
        print(f"Submission saved with {len(submission)} predictions")
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please update the file paths in the main() function")
    except Exception as e:
        print(f"Error running pipeline: {e}")

if __name__ == "__main__":
    main()
