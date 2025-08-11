#!/usr/bin/env python
# coding: utf-8

"""
Optimized Sticker Price Prediction Model
========================================
This script implements an optimized machine learning pipeline for sticker price prediction
with advanced feature engineering, hyperparameter tuning, and model evaluation.
"""

# Import Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging
from typing import Tuple, Dict, Any, List
import warnings

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

# Additional libraries
import holidays
import plotnine as p9 
from plotnine import *

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class for better parameter management
class Config:
    def __init__(self):
        # File paths - make configurable
        self.data_dir = Path("data")
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        self.output_dir = Path("output")
        
        # Model parameters
        self.test_size = 0.25
        self.random_state = 42
        self.cv_folds = 5
        
        # Feature engineering parameters
        self.lag_periods = [1, 7, 14, 30]
        self.rolling_windows = [7, 14, 30]
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        
    def get_file_paths(self):
        """Get file paths, fall back to original hardcoded paths if config paths don't exist"""
        if self.train_path.exists() and self.test_path.exists():
            return str(self.train_path), str(self.test_path)
        else:
            # Fallback to original paths
            return r"Z:\Sasindu\Data set\Sticker_Sales\train.csv", r"Z:\Sasindu\Data set\Sticker_Sales\test.csv"

config = Config()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess initial data
    
    Returns:
        Tuple of train and test DataFrames
    """
    train_path, test_path = config.get_file_paths()
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Data loaded successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Load data
train_df, test_df = load_data()


def initial_data_exploration(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Perform initial data exploration and preprocessing
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    """
    logger.info("Starting initial data exploration...")
    
    # Convert date columns
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # Log basic information
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    logger.info(f"Training data null values: {train_df.isnull().sum().sum()}")
    logger.info(f"Test data null values: {test_df.isnull().sum().sum()}")
    
    # Print unique value counts for categorical columns
    for col in train_df.columns:
        if train_df[col].dtype == 'object' or col in ['country', 'store', 'product']:
            logger.info(f"{col} unique values: {train_df[col].nunique()}")

# Perform initial exploration
initial_data_exploration(train_df, test_df)



def perform_eda(train_df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis with optimized visualizations
    
    Args:
        train_df: Training DataFrame
    """
    logger.info("Performing exploratory data analysis...")
    
    categorical_columns = ['country', 'store', 'product']
    
    # Basic statistics
    for col in categorical_columns:
        logger.info(f"{col} value counts: {train_df[col].value_counts().to_dict()}")
        logger.info(f"{col} mean num_sold: {train_df.groupby(col)['num_sold'].mean().to_dict()}")

def optimize_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized holiday feature engineering using vectorized operations
    
    Args:
        df: DataFrame with date and country columns
        
    Returns:
        DataFrame with holiday features added
    """
    logger.info("Engineering holiday features...")
    
    # Initialize holiday column
    df = df.copy()
    df["holiday"] = 0
    
    # Get unique countries and their holiday calendars
    country_mapping = {
        "Canada": holidays.country_holidays('CA'),
        "Finland": holidays.country_holidays('FI'),
        "Italy": holidays.country_holidays('IT'),
        "Kenya": holidays.country_holidays('KE'),
        "Norway": holidays.country_holidays('NO'),
        "Singapore": holidays.country_holidays('SG')
    }
    
    # Vectorized holiday detection
    for country, holiday_calendar in country_mapping.items():
        country_mask = df["country"] == country
        date_in_holidays = df.loc[country_mask, "date"].isin(holiday_calendar)
        df.loc[country_mask & date_in_holidays, "holiday"] = 1
    
    logger.info(f"Holiday feature added. Holiday count: {df['holiday'].sum()}")
    return df

def engineer_advanced_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer advanced time-based features including lag and rolling features
    
    Args:
        df: DataFrame with date column
        
    Returns:
        DataFrame with advanced time features
    """
    logger.info("Engineering advanced time features...")
    
    df = df.copy()
    
    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Sort by date for lag features
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add lag features if target exists (for training data)
    if 'num_sold' in df.columns:
        for lag in config.lag_periods:
            df[f'num_sold_lag_{lag}'] = df.groupby(['country', 'store', 'product'])['num_sold'].shift(lag)
        
        # Add rolling window statistics
        for window in config.rolling_windows:
            df[f'num_sold_rolling_mean_{window}'] = df.groupby(['country', 'store', 'product'])['num_sold'].rolling(window=window, min_periods=1).mean().values
            df[f'num_sold_rolling_std_{window}'] = df.groupby(['country', 'store', 'product'])['num_sold'].rolling(window=window, min_periods=1).std().values
    
    logger.info(f"Advanced time features engineered. New shape: {df.shape}")
    return df

def optimize_periodic_transform(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """
    Optimized periodic transformation using vectorized operations
    
    Args:
        df: DataFrame
        variables: List of variables to transform
        
    Returns:
        DataFrame with periodic features
    """
    logger.info("Applying periodic transformations...")
    
    df = df.copy()
    
    for variable in variables:
        if variable in df.columns:
            max_val = df[variable].max()
            df[f"{variable}_SIN"] = np.sin(df[variable] / max_val * 2 * np.pi)
            df[f"{variable}_COS"] = np.cos(df[variable] / max_val * 2 * np.pi)
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with improved strategy
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    df = df.copy()
    
    # For numerical columns, use group-based imputation
    if 'num_sold' in df.columns:
        # Group-based mean imputation for target variable
        df['num_sold'] = df.groupby('country')['num_sold'].transform(lambda x: x.fillna(x.mean()))
        
        # If still missing, use overall mean
        df['num_sold'] = df['num_sold'].fillna(df['num_sold'].mean())
    
    logger.info(f"Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
    return df


def feature_selection_pipeline(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, method: str = 'combined') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive feature selection pipeline
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        method: Feature selection method ('correlation', 'univariate', 'rfe', 'combined')
        
    Returns:
        Selected training and test features
    """
    logger.info(f"Starting feature selection using {method} method...")
    
    if method == 'correlation':
        # Remove highly correlated features
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        X_train_selected = X_train.drop(columns=high_corr_features)
        X_test_selected = X_test.drop(columns=high_corr_features) if X_test is not None else None
        
    elif method == 'univariate':
        # Univariate feature selection
        k_best = min(50, X_train.shape[1])  # Select top k features or all if less than k
        selector = SelectKBest(score_func=f_regression, k=k_best)
        X_train_selected = pd.DataFrame(selector.fit_transform(X_train, y_train), 
                                      columns=X_train.columns[selector.get_support()],
                                      index=X_train.index)
        
        if X_test is not None:
            X_test_selected = pd.DataFrame(selector.transform(X_test), 
                                         columns=X_train.columns[selector.get_support()],
                                         index=X_test.index)
        else:
            X_test_selected = None
            
    elif method == 'rfe':
        # Recursive Feature Elimination
        rf_temp = RandomForestRegressor(n_estimators=10, random_state=config.random_state)
        n_features = min(30, X_train.shape[1])  # Select top n features
        selector = RFE(rf_temp, n_features_to_select=n_features, step=1)
        
        X_train_selected = pd.DataFrame(selector.fit_transform(X_train, y_train), 
                                      columns=X_train.columns[selector.get_support()],
                                      index=X_train.index)
        
        if X_test is not None:
            X_test_selected = pd.DataFrame(selector.transform(X_test), 
                                         columns=X_train.columns[selector.get_support()],
                                         index=X_test.index)
        else:
            X_test_selected = None
            
    elif method == 'combined':
        # Combined approach: correlation + univariate
        # First remove highly correlated features
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        X_temp = X_train.drop(columns=high_corr_features)
        
        # Then apply univariate selection
        k_best = min(40, X_temp.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_best)
        X_train_selected = pd.DataFrame(selector.fit_transform(X_temp, y_train), 
                                      columns=X_temp.columns[selector.get_support()],
                                      index=X_train.index)
        
        if X_test is not None:
            X_test_temp = X_test.drop(columns=high_corr_features)
            X_test_selected = pd.DataFrame(selector.transform(X_test_temp), 
                                         columns=X_temp.columns[selector.get_support()],
                                         index=X_test.index)
        else:
            X_test_selected = None
    
    logger.info(f"Feature selection completed. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete feature preparation pipeline
    
    Args:
        train_df: Raw training data
        test_df: Raw test data
        
    Returns:
        Processed train and test DataFrames with features and targets separated
    """
    logger.info("Starting complete feature preparation pipeline...")
    
    # Handle missing values
    train_df = handle_missing_values(train_df)
    
    # Add holiday features
    train_df = optimize_holiday_features(train_df)
    test_df = optimize_holiday_features(test_df)
    
    # Add advanced time features
    train_df = engineer_advanced_time_features(train_df)
    test_df = engineer_advanced_time_features(test_df)
    
    # One-hot encoding
    categorical_columns = ['country', 'store', 'product']
    train_encoded = pd.get_dummies(train_df, columns=categorical_columns)
    test_encoded = pd.get_dummies(test_df, columns=categorical_columns)
    
    # Ensure both datasets have same columns
    all_columns = set(train_encoded.columns) | set(test_encoded.columns)
    for col in all_columns:
        if col not in train_encoded.columns:
            train_encoded[col] = 0
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    
    # Reorder columns
    common_columns = sorted([col for col in all_columns if col != 'num_sold'])
    train_encoded = train_encoded[common_columns + (['num_sold'] if 'num_sold' in train_encoded.columns else [])]
    test_encoded = test_encoded[common_columns]
    
    # Apply periodic transformations
    cyclic_columns = ['month', 'day', 'day_of_week', 'quarter', 'week_of_year']
    train_encoded = optimize_periodic_transform(train_encoded, cyclic_columns)
    test_encoded = optimize_periodic_transform(test_encoded, cyclic_columns)
    
    # Drop original cyclic columns and unnecessary columns
    drop_columns = ['date', 'id'] + cyclic_columns
    drop_columns = [col for col in drop_columns if col in train_encoded.columns]
    train_final = train_encoded.drop(columns=drop_columns)
    
    drop_columns_test = [col for col in drop_columns if col in test_encoded.columns]
    test_final = test_encoded.drop(columns=drop_columns_test)
    
    # Separate features and target
    if 'num_sold' in train_final.columns:
        X_train = train_final.drop(columns=['num_sold'])
        y_train = train_final['num_sold']
    else:
        X_train = train_final
        y_train = None
    
    X_test = test_final
    
    logger.info(f"Feature preparation completed. Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, test_encoded

# Apply complete feature preparation pipeline
X_train, X_test_final, y_train, test_encoded_with_id = prepare_features(train_df, test_df)

# Perform EDA on processed data
perform_eda(train_df)

# Apply feature selection
X_train_selected, X_test_selected = feature_selection_pipeline(X_train, y_train, X_test_final, method='combined')

logger.info("Feature engineering and selection completed.")


def create_model_with_tuning():
    """
    Create optimized models with hyperparameter tuning
    
    Returns:
        Dictionary of tuned models
    """
    logger.info("Creating models with hyperparameter tuning...")
    
    # Random Forest with hyperparameter tuning
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=config.random_state)
    rf_tuned = RandomizedSearchCV(
        rf, rf_param_grid, 
        n_iter=20, 
        cv=config.cv_folds, 
        scoring='neg_mean_absolute_error',
        random_state=config.random_state,
        n_jobs=-1
    )
    
    # XGBoost with hyperparameter tuning
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'alpha': [0, 0.1, 1, 10],
        'lambda': [1, 2, 5, 10]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=config.random_state)
    xgb_tuned = RandomizedSearchCV(
        xgb_model, xgb_param_grid,
        n_iter=20,
        cv=config.cv_folds,
        scoring='neg_mean_absolute_error',
        random_state=config.random_state,
        n_jobs=-1
    )
    
    # Decision Tree with hyperparameter tuning
    dt_param_grid = {
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['squared_error', 'absolute_error']
    }
    
    dt = DecisionTreeRegressor(random_state=config.random_state)
    dt_tuned = GridSearchCV(
        dt, dt_param_grid,
        cv=config.cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    models = {
        'Random Forest': rf_tuned,
        'XGBoost': xgb_tuned,
        'Decision Tree': dt_tuned
    }
    
    return models

def evaluate_model_with_cv(model, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Dict[str, float]:
    """
    Evaluate model using cross-validation
    
    Args:
        model: Model to evaluate
        X_train: Training features
        y_train: Training target
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} with cross-validation...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=config.cv_folds, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Additional metrics using train-test split
    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
        X_train, y_train, test_size=config.test_size, random_state=config.random_state
    )
    
    model.fit(X_temp_train, y_temp_train)
    y_pred = model.predict(X_temp_test)
    
    metrics = {
        'CV_MAE_mean': cv_mae,
        'CV_MAE_std': cv_std,
        'MAE': mean_absolute_error(y_temp_test, y_pred),
        'MSE': mean_squared_error(y_temp_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_temp_test, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_temp_test, y_pred),
        'R2': r2_score(y_temp_test, y_pred)
    }
    
    logger.info(f"{model_name} CV MAE: {cv_mae:.4f} (±{cv_std:.4f})")
    return metrics

def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train and evaluate all models with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dictionary with trained models and their metrics
    """
    logger.info("Training and evaluating models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    models = create_model_with_tuning()
    results = {}
    
    for model_name, model in models.items():
        try:
            # Fit the model (hyperparameter tuning happens here)
            model.fit(X_train_scaled_df, y_train)
            
            # Evaluate the best model
            best_model = model.best_estimator_
            metrics = evaluate_model_with_cv(best_model, X_train_scaled_df, y_train, model_name)
            
            results[model_name] = {
                'model': model,
                'best_model': best_model,
                'best_params': model.best_params_,
                'metrics': metrics,
                'scaler': scaler
            }
            
            logger.info(f"{model_name} - Best params: {model.best_params_}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def create_ensemble_model(models_dict: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> VotingRegressor:
    """
    Create an ensemble model using the best individual models
    
    Args:
        models_dict: Dictionary of trained models
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained ensemble model
    """
    logger.info("Creating ensemble model...")
    
    # Get best models for ensembling
    estimators = []
    for name, model_info in models_dict.items():
        if 'best_model' in model_info:
            estimators.append((name.lower().replace(' ', '_'), model_info['best_model']))
    
    if len(estimators) >= 2:
        ensemble = VotingRegressor(estimators=estimators)
        
        # Use the same scaler as individual models
        scaler = list(models_dict.values())[0]['scaler']
        X_train_scaled = scaler.transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        ensemble.fit(X_train_scaled_df, y_train)
        
        # Evaluate ensemble
        metrics = evaluate_model_with_cv(ensemble, X_train_scaled_df, y_train, 'Ensemble')
        
        logger.info(f"Ensemble model created with {len(estimators)} estimators")
        return ensemble, metrics, scaler
    else:
        logger.warning("Not enough models for ensemble. Need at least 2 models.")
        return None, None, None

# Split data for training and validation
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    X_train_selected, y_train, test_size=config.test_size, random_state=config.random_state
)

# Train and evaluate models
model_results = train_and_evaluate_models(X_train_selected, y_train)

# Create ensemble model
ensemble_model, ensemble_metrics, ensemble_scaler = create_ensemble_model(model_results, X_train_selected, y_train)

logger.info("Model training and evaluation completed.")


def compare_models(model_results: Dict[str, Any], ensemble_metrics: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compare all models and create a comprehensive comparison table
    
    Args:
        model_results: Dictionary with model results
        ensemble_metrics: Ensemble model metrics
        
    Returns:
        DataFrame with model comparison
    """
    logger.info("Comparing model performance...")
    
    comparison_data = []
    
    # Individual models
    for model_name, model_info in model_results.items():
        metrics = model_info['metrics']
        comparison_data.append({
            'Model': model_name,
            'CV_MAE': metrics['CV_MAE_mean'],
            'CV_MAE_Std': metrics['CV_MAE_std'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2']
        })
    
    # Ensemble model
    if ensemble_metrics:
        comparison_data.append({
            'Model': 'Ensemble',
            'CV_MAE': ensemble_metrics['CV_MAE_mean'],
            'CV_MAE_Std': ensemble_metrics['CV_MAE_std'],
            'MAE': ensemble_metrics['MAE'],
            'MSE': ensemble_metrics['MSE'],
            'RMSE': ensemble_metrics['RMSE'],
            'MAPE': ensemble_metrics['MAPE'],
            'R2': ensemble_metrics['R2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('CV_MAE').reset_index(drop=True)
    
    logger.info("Model comparison completed")
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return comparison_df

def visualize_predictions(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
    """
    Visualize predictions from different models
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
    """
    logger.info("Creating prediction visualizations...")
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, model_info) in enumerate(models_dict.items()):
        if idx >= len(axes):
            break
            
        # Get predictions
        scaler = model_info['scaler']
        X_test_scaled = scaler.transform(X_test)
        best_model = model_info['best_model']
        y_pred = best_model.predict(X_test_scaled)
        
        # Plot predictions vs actual
        axes[idx].scatter(y_test, y_pred, alpha=0.6, s=1)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[idx].set_xlabel('Actual')
        axes[idx].set_ylabel('Predicted')
        axes[idx].set_title(f'{model_name}\nMAPE: {model_info["metrics"]["MAPE"]:.4f}')
    
    # Remove empty subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(config.output_dir / 'model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model_results: Dict[str, Any], feature_names: List[str]):
    """
    Analyze and visualize feature importance from different models
    
    Args:
        model_results: Dictionary of trained models
        feature_names: List of feature names
    """
    logger.info("Analyzing feature importance...")
    
    importance_data = {}
    
    for model_name, model_info in model_results.items():
        best_model = model_info['best_model']
        
        if hasattr(best_model, 'feature_importances_'):
            importance_data[model_name] = best_model.feature_importances_
    
    if importance_data:
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        
        # Plot feature importance
        fig, axes = plt.subplots(1, len(importance_data), figsize=(20, 8))
        if len(importance_data) == 1:
            axes = [axes]
        
        for idx, (model_name, importances) in enumerate(importance_data.items()):
            top_features = importance_df[model_name].nlargest(15)
            
            axes[idx].barh(range(len(top_features)), top_features.values)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features.index)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name} - Top 15 Features')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(config.output_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    return None

def select_best_model(model_results: Dict[str, Any], ensemble_model, ensemble_metrics: Dict[str, float]) -> Tuple[Any, str, Any]:
    """
    Select the best performing model based on cross-validation MAE
    
    Args:
        model_results: Dictionary of individual models
        ensemble_model: Ensemble model
        ensemble_metrics: Ensemble metrics
        
    Returns:
        Best model, model name, and associated scaler
    """
    logger.info("Selecting best model...")
    
    best_mae = float('inf')
    best_model = None
    best_name = None
    best_scaler = None
    
    # Check individual models
    for model_name, model_info in model_results.items():
        cv_mae = model_info['metrics']['CV_MAE_mean']
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_model = model_info['best_model']
            best_name = model_name
            best_scaler = model_info['scaler']
    
    # Check ensemble model
    if ensemble_metrics and ensemble_metrics['CV_MAE_mean'] < best_mae:
        best_model = ensemble_model
        best_name = 'Ensemble'
        best_scaler = ensemble_scaler
        best_mae = ensemble_metrics['CV_MAE_mean']
    
    logger.info(f"Best model selected: {best_name} with CV MAE: {best_mae:.4f}")
    return best_model, best_name, best_scaler

def generate_final_predictions(best_model, best_scaler, X_test_final: pd.DataFrame, test_encoded_with_id: pd.DataFrame) -> pd.DataFrame:
    """
    Generate final predictions for submission
    
    Args:
        best_model: Best performing model
        best_scaler: Scaler used for the best model
        X_test_final: Final test features
        test_encoded_with_id: Test data with ID column
        
    Returns:
        Submission DataFrame
    """
    logger.info("Generating final predictions...")
    
    # Scale test features
    X_test_scaled = best_scaler.transform(X_test_final)
    
    # Make predictions
    predictions = best_model.predict(X_test_scaled)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_encoded_with_id['id'] if 'id' in test_encoded_with_id.columns else range(len(predictions)),
        'num_sold': predictions
    })
    
    # Save to file
    submission_path = config.output_dir / 'optimized_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"Final predictions saved to {submission_path}")
    logger.info(f"Prediction statistics - Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")
    
    return submission_df

# Compare all models
comparison_df = compare_models(model_results, ensemble_metrics)

# Visualize predictions and analyze feature importance
if model_results:
    visualize_predictions(model_results, X_temp_test, y_temp_test)
    feature_importance_df = analyze_feature_importance(model_results, X_train_selected.columns.tolist())

# Select the best model
best_model, best_model_name, best_scaler = select_best_model(model_results, ensemble_model, ensemble_metrics)

# Generate final predictions
final_submission = generate_final_predictions(best_model, best_scaler, X_test_selected, test_encoded_with_id)

print(f"\n🎉 OPTIMIZATION COMPLETE!")
print(f"Best performing model: {best_model_name}")
print(f"Final submission shape: {final_submission.shape}")
print(f"Predictions saved to: {config.output_dir / 'optimized_submission.csv'}")

logger.info("All optimizations completed successfully!")

