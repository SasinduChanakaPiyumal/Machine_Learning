#!/usr/bin/env python
"""
Micro-benchmark script to measure performance of sticker sales prediction pipeline
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from functools import wraps
from typing import Callable, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def benchmark_function(func: Callable) -> Callable:
    """Decorator to benchmark function execution time and memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the function
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = final_memory - initial_memory
        
        execution_time = end_time - start_time
        print(f"{func.__name__}: {execution_time:.4f}s, Memory: {memory_diff:.2f}MB")
        
        return result, execution_time, memory_diff
    return wrapper

def create_sample_data(n_rows: int = 10000) -> tuple:
    """Create sample data for benchmarking"""
    countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
    stores = ['Discount Stickers', 'Premium Stickers', 'Online Store']
    products = ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']
    
    # Generate date range
    date_range = pd.date_range(start='2010-01-01', end='2016-12-31', freq='D')
    
    train_data = pd.DataFrame({
        'id': range(n_rows),
        'date': np.random.choice(date_range, n_rows),
        'country': np.random.choice(countries, n_rows),
        'store': np.random.choice(stores, n_rows),
        'product': np.random.choice(products, n_rows),
        'num_sold': np.random.normal(500, 200, n_rows)
    })
    
    test_data = pd.DataFrame({
        'id': range(n_rows, n_rows + 1000),
        'date': np.random.choice(pd.date_range(start='2017-01-01', end='2017-12-31', freq='D'), 1000),
        'country': np.random.choice(countries, 1000),
        'store': np.random.choice(stores, 1000),
        'product': np.random.choice(products, 1000)
    })
    
    return train_data, test_data

@benchmark_function
def original_holiday_processing(df):
    """Original inefficient holiday processing function"""
    df = df.copy()
    df["holiday"] = 0
    
    # Create holiday objects
    ca_holidays = holidays.country_holidays('CA')
    fi_holidays = holidays.country_holidays('FI')
    it_holidays = holidays.country_holidays('IT')
    ke_holidays = holidays.country_holidays('KE')
    no_holidays = holidays.country_holidays('NO')
    sg_holidays = holidays.country_holidays('SG')
    
    def set_holiday(row):
        VAL_HOLIDAY = 1
        if row["country"] == "Canada" and row["date"] in ca_holidays:
            row["holiday"] = VAL_HOLIDAY
        elif row["country"] == "Finland" and row["date"] in fi_holidays:
            row["holiday"] = VAL_HOLIDAY
        elif row["country"] == "Italy" and row["date"] in it_holidays:
            row["holiday"] = VAL_HOLIDAY
        elif row["country"] == "Kenya" and row["date"] in ke_holidays:
            row["holiday"] = VAL_HOLIDAY
        elif row["country"] == "Norway" and row["date"] in no_holidays:
            row["holiday"] = VAL_HOLIDAY
        elif row["country"] == "Singapore" and row["date"] in sg_holidays:
            row["holiday"] = VAL_HOLIDAY
        return row
    
    # This is the bottleneck - row-by-row processing
    df_result = df.apply(set_holiday, axis=1)
    return df_result

@benchmark_function
def optimized_holiday_processing(df):
    """Optimized vectorized holiday processing function"""
    df = df.copy()
    df["holiday"] = 0
    
    # Create holiday objects once and cache them
    holiday_cache = {
        'Canada': holidays.country_holidays('CA'),
        'Finland': holidays.country_holidays('FI'),
        'Italy': holidays.country_holidays('IT'),
        'Kenya': holidays.country_holidays('KE'),
        'Norway': holidays.country_holidays('NO'),
        'Singapore': holidays.country_holidays('SG')
    }
    
    # Vectorized approach using boolean indexing
    for country, holiday_obj in holiday_cache.items():
        country_mask = df['country'] == country
        date_mask = df.loc[country_mask, 'date'].isin(holiday_obj.keys())
        df.loc[country_mask & date_mask, 'holiday'] = 1
    
    return df

@benchmark_function
def original_periodic_transform(df):
    """Original inefficient periodic transformation"""
    df = df.copy()
    
    def periodic_transform(dff, variable):
        dff[f"{variable}_SIN"] = np.sin(dff[variable] / dff[variable].max() * 2 * np.pi)
        dff[f"{variable}_COS"] = np.cos(dff[variable] / dff[variable].max() * 2 * np.pi)
        return dff
    
    cyclic_col = ['month', 'day', 'day_of_week']
    
    for col in cyclic_col:
        df = periodic_transform(df, col)
    
    return df

@benchmark_function
def optimized_periodic_transform(df):
    """Optimized vectorized periodic transformation"""
    df = df.copy()
    
    cyclic_col = ['month', 'day', 'day_of_week']
    
    for col in cyclic_col:
        if col in df.columns:
            max_val = df[col].max()
            df[f"{col}_SIN"] = np.sin(df[col] / max_val * 2 * np.pi)
            df[f"{col}_COS"] = np.cos(df[col] / max_val * 2 * np.pi)
    
    return df

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing original vs optimized functions"""
    print("Creating sample data...")
    train_data, test_data = create_sample_data(n_rows=10000)
    
    # Add date features for periodic transform benchmark
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data['year'] = train_data['date'].dt.year
    train_data['month'] = train_data['date'].dt.month
    train_data['day'] = train_data['date'].dt.day
    train_data['day_of_week'] = train_data['date'].dt.dayofweek
    
    print("\n" + "="*60)
    print("HOLIDAY PROCESSING BENCHMARK")
    print("="*60)
    
    # Benchmark holiday processing
    print("\nOriginal holiday processing:")
    _, orig_time, orig_memory = original_holiday_processing(train_data)
    
    print("\nOptimized holiday processing:")
    _, opt_time, opt_memory = optimized_holiday_processing(train_data)
    
    holiday_speedup = orig_time / opt_time
    memory_savings = orig_memory - opt_memory
    
    print(f"\nHoliday Processing Results:")
    print(f"Speedup: {holiday_speedup:.2f}x faster")
    print(f"Memory savings: {memory_savings:.2f}MB")
    
    print("\n" + "="*60)
    print("PERIODIC TRANSFORM BENCHMARK")
    print("="*60)
    
    # Benchmark periodic transform
    print("\nOriginal periodic transform:")
    _, orig_time2, orig_memory2 = original_periodic_transform(train_data)
    
    print("\nOptimized periodic transform:")
    _, opt_time2, opt_memory2 = optimized_periodic_transform(train_data)
    
    periodic_speedup = orig_time2 / opt_time2
    memory_savings2 = orig_memory2 - opt_memory2
    
    print(f"\nPeriodic Transform Results:")
    print(f"Speedup: {periodic_speedup:.2f}x faster")
    print(f"Memory savings: {memory_savings2:.2f}MB")
    
    # Overall results
    total_orig_time = orig_time + orig_time2
    total_opt_time = opt_time + opt_time2
    overall_speedup = total_orig_time / total_opt_time
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Total original time: {total_orig_time:.4f}s")
    print(f"Total optimized time: {total_opt_time:.4f}s")
    print(f"Overall speedup: {overall_speedup:.2f}x faster")
    print(f"Total memory savings: {(memory_savings + memory_savings2):.2f}MB")

if __name__ == "__main__":
    print("Starting Sticker Sales Performance Benchmark...")
    run_comprehensive_benchmark()
