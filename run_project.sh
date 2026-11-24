#!/bin/bash
# ==============================================================================
# GridCast AI — India Electricity Demand & Price Forecasting (2000-2025)
# All-In-One Project Runner
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================================="
echo "⚡ Starting GridCast AI — Electricity Demand & Price Forecasting"
echo "=========================================================================="
echo "Working directory: $PROJECT_DIR"
echo ""

# 1. Verify datasets
if [ ! -f "data/weather_features_india.csv" ] || [ ! -f "data/energy_dataset_india.csv" ]; then
    echo "[!] Datasets not found in data/. Generating fresh datasets..."
    python3 fetch_indian_data.py --live --start-year 2020 --end-year 2025
else
    echo "[✓] Indian weather and energy datasets verified in data/"
fi

# 2. Check model benchmark
if [ ! -f "data/model_benchmark_india.csv" ]; then
    echo "[*] Training and evaluating forecasting models on Indian data..."
    python3 train_india_models.py
else
    echo "[✓] Model benchmarks and predictions verified in data/"
fi

# 3. Launch interactive landing page
echo ""
echo "[*] Opening GridCast AI interactive landing page in your browser..."
open "landing/index.html" || python3 -m webbrowser "file://$PROJECT_DIR/landing/index.html"

echo ""
echo "=========================================================================="
echo "🎉 GridCast AI is now running on your machine!"
echo "   • Interactive Landing Page: file://$PROJECT_DIR/landing/index.html"
echo "   • Indian Jupyter Notebook: Electricity_Demand_and_Price_forecasting_India.ipynb"
echo "   • Benchmark Metrics:       data/model_benchmark_india.csv"
echo "   • Test Set Predictions:    data/model_predictions_india.csv"
echo "=========================================================================="
