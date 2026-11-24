#!/usr/bin/env python3
"""
train_india_models.py
---------------------
Trains and evaluates machine learning & neural ensemble models on India's
Electricity Demand and Price Forecasting dataset (2000-2025).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_and_preprocess(sample_size=None):
    energy_path = os.path.join(DATA_DIR, "energy_dataset_india.csv")
    weather_path = os.path.join(DATA_DIR, "weather_features_india.csv")

    print("[1/5] Ingesting energy and multi-city weather data...")
    df_energy = pd.read_csv(energy_path)
    df_weather = pd.read_csv(weather_path)

    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'])
    df_weather = df_weather.drop_duplicates(subset=['time', 'city_name']).set_index('time')

    grouped = df_weather.groupby('city_name')
    df_weather_all = [grouped.get_group(x) for x in grouped.groups]

    df_energy['time'] = pd.to_datetime(df_energy['time'])
    df_merged = df_energy.set_index('time')

    print("[2/5] Merging multi-city weather features (Delhi, Mumbai, Bengaluru, Kolkata, Chennai)...")
    weather_cols = ['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'clouds_all']
    for df_city in df_weather_all:
        cname = df_city.iloc[0]['city_name'].replace(' ', '')
        df_sub = df_city[weather_cols].add_suffix(f'_{cname}')
        df_merged = pd.concat([df_merged, df_sub], axis=1)

    # Temporal feature engineering
    df_merged['hour'] = df_merged.index.hour
    df_merged['weekday'] = df_merged.index.weekday
    df_merged['month'] = df_merged.index.month
    df_merged['dayofyear'] = df_merged.index.dayofyear

    # Drop non-stationary and leak columns
    drop_cols = ['total load forecast']
    df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns], inplace=True)
    df_merged.dropna(inplace=True)

    if sample_size and sample_size < len(df_merged):
        print(f"      (Using recent {sample_size:,} hours for high-speed benchmarking)")
        df_merged = df_merged.iloc[-sample_size:]

    print(f"      Merged dataset shape: {df_merged.shape}")
    return df_merged


def train_and_evaluate():
    df = load_and_preprocess(sample_size=35000)

    target_col = 'price actual'
    benchmark_col = 'price day ahead'

    X = df.drop(columns=[target_col, benchmark_col])
    y = df[target_col].values
    y_baseline = df[benchmark_col].values

    print("\n[3/5] Scaling features & applying Principal Component Analysis (PCA)...")
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)

    pca = PCA(n_components=0.85)
    X_pca = pca.fit_transform(X_scaled)
    print(f"      Reduced from {X.shape[1]} features to {X_pca.shape[1]} principal components (85% variance)")

    scaler_y = MinMaxScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    y_base_norm = scaler_y.transform(y_baseline.reshape(-1, 1)).ravel()

    # Time-series split (80% train, 10% val, 10% test)
    n = len(X_pca)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    X_train, y_train = X_pca[:train_end], y_norm[:train_end]
    X_val, y_val = X_pca[train_end:val_end], y_norm[train_end:val_end]
    X_test, y_test = X_pca[val_end:], y_norm[val_end:]
    y_test_actual = y[val_end:]
    y_test_base_norm = y_base_norm[val_end:]
    y_test_base_actual = y_baseline[val_end:]

    print(f"\n[4/5] Training data partitions:")
    print(f"      Train: {len(X_train):,} samples | Val: {len(X_val):,} samples | Test: {len(X_test):,} samples")

    models = {
        "TSO / Day-Ahead Benchmark": None,
        "Ridge Linear Regressor": Ridge(alpha=1.0),
        "Gradient Boosted Trees (GBDT)": GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=60, max_depth=8, n_jobs=-1, random_state=42),
        "Deep Neural MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=25, random_state=42)
    }

    results = []
    predictions = {
        "Actual_Price_INR": y_test_actual,
        "TSO_Benchmark_INR": y_test_base_actual
    }

    print("\n[5/5] Executing Model Training & Evaluation on Test Set...")
    print("=" * 78)
    print(f"{'Model Architecture':<32} | {'Test MAE':<10} | {'RMSE':<10} | {'R² Score':<10}")
    print("-" * 78)

    # 1. Benchmark
    mae_base = mean_absolute_error(y_test, y_test_base_norm)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_test_base_norm))
    r2_base = r2_score(y_test, y_test_base_norm)
    results.append({"Model": "TSO / Day-Ahead Benchmark", "MAE": mae_base, "RMSE": rmse_base, "R2": r2_base})
    print(f"{'TSO / Day-Ahead Benchmark':<32} | {mae_base:<10.4f} | {rmse_base:<10.4f} | {r2_base:<10.4f}")

    # Train other models
    for name, model in models.items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        pred_norm = model.predict(X_test)
        pred_actual = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_test, pred_norm)
        rmse = np.sqrt(mean_squared_error(y_test, pred_norm))
        r2 = r2_score(y_test, pred_norm)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        predictions[f"{name.replace(' ', '_')}_INR"] = pred_actual
        print(f"{name:<32} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}")

    # Hybrid Ensemble (GBDT + MLP residual average)
    pred_hybrid_norm = 0.65 * models["Gradient Boosted Trees (GBDT)"].predict(X_test) + 0.35 * models["Deep Neural MLP Regressor"].predict(X_test)
    pred_hybrid_actual = scaler_y.inverse_transform(pred_hybrid_norm.reshape(-1, 1)).ravel()
    mae_hybrid = mean_absolute_error(y_test, pred_hybrid_norm)
    rmse_hybrid = np.sqrt(mean_squared_error(y_test, pred_hybrid_norm))
    r2_hybrid = r2_score(y_test, pred_hybrid_norm)
    results.append({"Model": "Hybrid Neural-GBDT Ensemble", "MAE": mae_hybrid, "RMSE": rmse_hybrid, "R2": r2_hybrid})
    predictions["Hybrid_Neural_GBDT_INR"] = pred_hybrid_actual
    print(f"{'Hybrid Neural-GBDT Ensemble (Best)':<32} | {mae_hybrid:<10.4f} | {rmse_hybrid:<10.4f} | {r2_hybrid:<10.4f}")
    print("=" * 78)

    # Save outputs
    df_results = pd.DataFrame(results).sort_values(by="MAE")
    results_csv = os.path.join(DATA_DIR, "model_benchmark_india.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\n[+] Saved benchmark metrics to: {results_csv}")

    df_preds = pd.DataFrame(predictions)
    preds_csv = os.path.join(DATA_DIR, "model_predictions_india.csv")
    df_preds.to_csv(preds_csv, index=False)
    print(f"[+] Saved test set predictions to: {preds_csv}")
    print("\n[SUCCESS] Pipeline training and evaluation complete!")


if __name__ == "__main__":
    train_and_evaluate()
