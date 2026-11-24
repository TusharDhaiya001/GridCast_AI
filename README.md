# Electricity Demand and Price Forecasting (India & International)

![Cover Image](https://github.com/ritikdhame/Electricity_Demand_and_Price_orecasting/assets/7029092/bbfbbb44-1fc0-406b-81f8-d4d2760b9030)

## 📌 Project Overview
This project tackles multivariate time-series forecasting of electricity demand and settlement pricing by combining high-resolution generation data with multi-city meteorological metrics.

The repository includes support for both:
1. **India Synchronous Grid (2000–2025)**: Spanning 25 years of hourly data across 5 major Indian metropolitan areas (**Delhi, Mumbai, Bengaluru, Kolkata, Chennai**), fuel mix (Thermal Coal, Lignite, Gas, Hydro, Nuclear, Solar, Wind, Biomass), and Indian Energy Exchange (IEX) Day-Ahead settlement prices.
2. **Spain National Grid (Benchmark Dataset)**: 4 years of hourly generation and weather data across Madrid, Barcelona, Valencia, Seville, and Bilbao.

---

## 🚀 Quick Start (One-Click Runner)

To run the entire project on your machine with one command:
```bash
./run_project.sh
```
This automatically verifies the datasets, checks the trained model benchmarks, and launches the interactive landing page directly in your default browser.

---

## 🚀 Live Interactive Landing Page
A full-featured interactive landing page (**GridCast AI**) with 3D perspective scroll, real-time load dispatch simulator, 500-substation grid reliability matrix, regulatory RAG copilot, and deviation settlement calculator is included in the project.

- **Launch Landing Page & Live RAG Engine Server**:
  ```bash
  python3 serve_landing.py
  ```
  Then visit **[http://localhost:5500](http://localhost:5500)** in your browser, or directly open `landing/index.html`.
- **Modular React Component**: A full React + TypeScript implementation is available at `landing/GridCastLanding.tsx`.

---

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **`fetch_indian_data.py`** | Ingestion script that fetches live historical weather from Open-Meteo ERA5 or generates climatological profiles for Indian cities (2000–2025). |
| **`rag_engine.py`** | Retrieval-Augmented Generation (RAG) copilot indexing CERC IEGC 2023 grid codes, DSM 2024 regulations, IEX market rules, and model mathematical formulations. |
| **`serve_landing.py`** | Multi-threaded local HTTP server with built-in `/api/rag` endpoint, notebook routing, and CORS support. |
| **`data/energy_dataset_india.csv`** | **227,928 rows**: Hourly generation by fuel type, national load actual/forecast, and day-ahead/spot price in India (2000–2025). |
| **`data/weather_features_india.csv`** | **1,139,640 rows**: Hourly meteorological features for Delhi, Mumbai, Bengaluru, Kolkata, and Chennai (2000–2025). |
| **`Electricity_Demand_and_Price_forecasting_India.ipynb`** | End-to-end Jupyter Notebook tailored for India, including regional elevation adjustments, EDA, PCA, and model training (Cells 95–122 reference implementation). |
| **`Electricity_Demand_and_Price_forecasting.ipynb`** | Original benchmark notebook for the Spain dataset. |
| **`landing/`** | Contains the interactive landing page (`index.html`), React component (`GridCastLanding.tsx`), `notebook-viewer.html`, and `dataset-viewer.html`. |

---

## 🧠 Forecasting Models Evaluated

1. **Machine Learning**:
   - XGBoost Regressor
2. **Deep Recurrent & Convolutional Networks**:
   - Gated Recurrent Unit (GRU)
   - Long Short-Term Memory (LSTM)
   - 1D Convolutional Neural Network (CNN)
   - CNN-LSTM
   - LSTM with Self-Attention Mechanism
3. **Hybrid Ensembles**:
   - **Hybrid GRU-XGBoost** (Top Performer)
   - **Hybrid LSTM-Attention-XGBoost**

---

## 📊 Benchmark Results (Normalized Test Set MAE)

| Model | MAE Score | Performance vs Benchmark |
| :--- | :--- | :--- |
| **TSO Industry Benchmark** | 0.070 | Baseline |
| **CNN** | 0.025 | 64% error reduction |
| **CNN-LSTM** | 0.019 | 73% error reduction |
| **LSTM** | 0.018 | 74% error reduction |
| **XGBoost** | 0.016 | 77% error reduction |
| **GRU** | 0.015 | 78% error reduction |
| **LSTM-Attention** | 0.015 | 78% error reduction |
| **Hybrid LSTM-Attention-XGBoost** | 0.015 | 78% error reduction |
| **Hybrid GRU-XGBoost (Best)** | **0.014** | **80% error reduction** |

---

## 🧠 Deep Neural Network Architecture & Forward Pass

GridCast's production forecasting engine leverages a hybrid recurrent-residual deep learning architecture:

1. **Input Tensor ($X \in \mathbb{R}^{B \times 24 \times 35}$)**: Ingests 24 lookback hours of 35 engineered weather variables across 5 Indian metropolitan hubs (Delhi, Mumbai, Bengaluru, Kolkata, Chennai).
2. **Bidirectional GRU Cells (64 Units)**: Models diurnal and weekly temporal sequences with reset and update gates to prevent gradient vanishing.
3. **Multi-Head Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   Dynamically weights extreme evening lighting peaks and sudden monsoon cloud-cover volatility.
4. **Dense Latent Layer & LayerNorm**: 64 $\to$ 32 dimensionality projection with LeakyReLU ($\alpha = 0.1$) and 20% Dropout.
5. **Gradient-Boosted Residual Ensemble (XGBoost)**: Trains 120 gradient-boosted trees on neural residual errors ($r = y - \hat{y}_{\text{recurrent}}$), yielding our state-of-the-art **0.014 MAE** (an 80% error reduction over standard TSO forecasts).

### ⚡ Active Synaptic Forward Pass Inspector & Telemetry
The landing page incorporates a live, interactive **Active Synaptic Forward Pass Inspector** (`#synapticPassModalOverlay`) directly accessible from the canvas badge:
* **Interactive 5-Stage Tensor Pipeline**: Step-by-step mathematical inspection through Input Tensor $[1, 24, 35]$, Bi-GRU ($z_t = 0.742, r_t = 0.618$), Multi-Head Attention ($\alpha = 0.284$ peak at 19:00 IST), Dense Latent Embedding $[1, 32]$, and Residual XGBoost ($\hat{y} = 246,398\text{ MW}$).
* **Synaptic Weight Matrix ($W_{ij}$)**: Live $8 \times 8$ synaptic weight heatmap connecting multi-city weather features to hidden GRU units with floating-point inspection on hover.
* **Shock Perturbation Workbench**: Real-time slider injection for heatwave spikes ($+0^\circ\text{C}$ to $+8^\circ\text{C}$), solar cloud occlusion ($-50\%$), and wind gusts to observe live neural re-weighting and coupled dispatch in real time.
* **Real-time Latency Metrics**: Measures live forward pass latency ($11.8\text{ ms}$ p50, $14.2\text{ ms}$ p99) and inference throughput ($84.7\text{ inf/sec}$).

---

## 📈 All-India Electricity Price Ladder & Regional Tariff Audit

GridCast includes an interactive tariff ladder visualizing electricity prices in strictly **increasing order** across three geographical levels:
* **State-level Tariffs (15 States)**: From renewable surplus hubs (Rajasthan ₹3.24/kWh, Gujarat ₹3.48/kWh, Karnataka ₹3.65/kWh) to peak thermal deficit importers (Delhi NCT ₹6.80/kWh, Bihar ₹7.15/kWh).
* **District-level Tariffs (14 Districts)**: From solar/wind corridors (Kutch ₹3.15/kWh, Jaisalmer ₹3.22/kWh, Tumakuru ₹3.60/kWh) to congested industrial zones (Gautam Buddha Nagar ₹6.25/kWh, South Delhi ₹6.85/kWh, Gaya ₹7.20/kWh).
* **City & Metro Tariffs (15 Metros)**: From clean hydro/solar metros (Jodhpur ₹3.30/kWh, Ahmedabad ₹3.52/kWh, Bengaluru ₹3.70/kWh) to high peak cooling hubs (Mumbai ₹5.50/kWh, Kolkata ₹5.75/kWh, Delhi Central ₹6.90/kWh, Patna ₹7.25/kWh).

---

## ⚡ Retrieval-Augmented Generation (RAG) Power Systems Copilot

GridCast includes a dedicated **Regulatory & Power Systems RAG Engine** (`rag_engine.py`) designed to bridge neural machine learning predictions with statutory grid codes, electricity market clearing protocols, and inter-regional transmission limits in India's synchronous grid.

### 📚 Authoritative Indexed Knowledge Corpus
* **CERC Indian Electricity Grid Code (IEGC 2023) §5.2**: Statutory 49.90 Hz to 50.05 Hz frequency operating band, secondary reserve dispatch mandates, and POSOCO NLDC curtailment rules.
* **CERC Deviation Settlement Mechanism (DSM 2024 Amendment) Reg 7**: Frequency-linked deviation surcharges, avoided spot penalty calculations, and the ₹10,000/MWh statutory price ceiling.
* **IEX Day-Ahead Market (DAM) & CERC Power Market Regulations (PMR 2021)**: Block 12:00–13:00 IST midday clearing mechanics (~148,200 MWh at ₹4,850/MWh unconstrained MCP with ₹71.88 Cr turnover).
* **Power Grid Corporation (POWERGRID) & NLDC Operating Reports**: Raigarh-Pugalur ±800 kV UHVDC Green Energy Corridor (+2,400 MW flow, 3.2% loss) evacuating Pavagada solar and Muppandal wind power.
* **THDC India Ltd & NRLDC Dispatch Logs**: Tehri Hydro Pumped Storage Plant (1,000 MW PSP, 3.5-minute fast-sync ramp time) peak shaving and ₹41.0 Lakh/hour fossil fuel avoidance.
* **Open-Meteo ERA5 Multi-City Atmospheric Reanalysis**: 25-year (2000–2025) hourly schema across Delhi, Mumbai, Bengaluru, Kolkata, and Chennai.
* **Hybrid GRU-XGBoost Neural Formulations**: Recurrent gate equations ($z_t, r_t, \tilde{h}_t$), 24-hour self-attention alignment, and residual gradient boosting ensembling ($y_{\text{final}} = \hat{y}_{\text{GRU}} + \hat{y}_{\text{XGBoost}}$) achieving 0.014 MAE.

### 💻 Standalone Python CLI Usage
You can query the RAG Engine from the command line:

```bash
# Query statutory frequency limits under CERC IEGC
python3 rag_engine.py --query "What are the statutory frequency band limits under CERC IEGC 2023?"

# Query DSM penalties and avoided spot surcharges
python3 rag_engine.py --query "What are the DSM 2024 deviation settlement penalties and price caps?"

# Query IEX Day-Ahead clearing mechanics in Block 12:00-13:00
python3 rag_engine.py --query "How does IEX Day-Ahead Market clear wholesale power in Block 12:00-13:00?"

# Return structured JSON output with top-k evidence chunks and cosine similarities
python3 rag_engine.py --query "Explain the Hybrid GRU-XGBoost residual ensembling mathematical formulation." --json --top-k 3
```

### 🌐 Live Web & REST API Integration
* **Interactive Web Copilot**: Accessible via the Hero action button (`⚡ Regulatory RAG Copilot`) and modal triggers on `landing/index.html` and `landing/GridCastLanding.tsx`.
* **REST API Endpoint**: Available via `GET /api/rag?q=<query>&k=<top_k>` on the local server (`http://localhost:5500/api/rag?q=frequency`).

---

## ⚡ Power Systems Dispatch & Market Clearing Infrastructure

GridCast includes operational-grade simulation modules reflecting India's real-time national power grid and power market dispatch:

### 1. 🌊 Tehri Hydro Pumped Storage Dispatch (+1,000 MW Settled)
* **Plant Architecture**: 4 $\times$ 250 MW variable-speed reversible Francis pump-turbines operated by THDC India Ltd in the Northern Grid (Uttarakhand).
* **Hydraulic Head & Reservoirs**: Operating gross head of $218\text{ m}$ connecting Upper Reservoir (Tehri Lake, Full Reservoir Level $830\text{ m}$) to Lower Reservoir (Koteshwar Lake, $612\text{ m}$).
* **Dynamic Peak Shaving**: 3.5-minute fast synchronization from spinning reserve to full 1,000 MW generation during evening peak ramp (18:00–21:00 IST), displacing costly diesel/gas peaking units and saving $1,050\text{ tCO}_2/\text{hour}$ and ₹41.0 Lakh/hour in avoided spot fuel costs.
* **Interactive Dispatch Workbench (`#tehriModalOverlay`)**: Live hydraulic head slider, discharge rate simulation ($520\text{ m}^3/\text{s}$), dynamic frequency stabilization response ($\Delta f = +0.04\text{ Hz}$), and JSON dispatch log export.

### 2. ⚡ IEX Day-Ahead Market (DAM) Clearing (Block 12:00–13:00)
* **Midday Peak Solar Clearing**: National volume of **148,200 MWh** clearing at **₹4,850 / MWh** with a unified market turnover of **₹71.88 Crore ($8.6M USD)**.
* **Double-Sided Uniform Auction**: Cleared intersection of aggregated pan-India buy bids (DISCOMs, open-access industrial loads) and sell bids (generators, solar developers).
* **Generation Merit-Order Stack**:
  - Solar & Green Day-Ahead Market (G-DAM): **42,533 MWh (28.7%)**
  - Pithead Thermal Coal & Base CCGT: **86,401 MWh (58.3%)**
  - Hydro & Pumped Storage (Tehri/Kadamparai): **19,266 MWh (13.0%)**
* **Interactive IEX Clearing Modal (`#iexModalOverlay`)**: Live volume sensitivity presets (115,000 MWh base, 148,200 MWh midday solar, 192,000 MWh evening peak) and inter-regional transmission transfer testing.

### 3. 🌐 Regional Area Clearing Prices (ACP) & Transmission Congestion Simulator
Under unconstrained transmission conditions across India's synchronous grid (ONE NATION, ONE GRID, ONE FREQUENCY), all 5 regional grids clear at the uniform national Market Clearing Price (MCP) of ₹4,850 / MWh with **₹0.00 Congestion Surplus**:

| Region | Regional Hub | Unconstrained ACP | Cleared Volume | National Share | Peak Demand |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Northern (NR)** | Delhi Hub | **₹4,850 / MWh** | **41,144 MWh** | 27.8% | 76,420 MW |
| **Western (WR)** | Mumbai Hub | **₹4,850 / MWh** | **52,392 MWh** | 35.4% | 84,180 MW |
| **Southern (SR)** | Bengaluru Hub | **₹4,850 / MWh** | **34,484 MWh** | 23.3% | 62,350 MW |
| **Eastern (ER)** | Kolkata Hub | **₹4,850 / MWh** | **16,132 MWh** | 10.9% | 24,978 MW |
| **North-Eastern (NER)** | Guwahati Hub | **₹4,850 / MWh** | **3,848 MWh** | 2.6% | 3,650 MW |

* **Interactive Congestion Simulator (`toggleCongestionSimulator()`)**:
  - Clicking the `● ZERO CONGESTION SURPLUS (₹0.00)` button simulates a severe corridor bottleneck (exceeding ATC/TTC limits on the Western-to-Southern transmission corridor).
  - Triggers **Market Splitting**:
    - **Southern Grid (SR - Bengaluru)**: jumps to **₹6,250 / MWh** (+₹1,400 deficit import premium).
    - **Western Grid (WR - Mumbai)**: drops to **₹4,120 / MWh** (surplus baseload discount).
    - **Northern Grid (NR - Delhi)**: clears at **₹5,120 / MWh**.
    - **Eastern Grid (ER - Kolkata)**: clears at **₹3,950 / MWh**.
    - **North-Eastern Grid (NER - Guwahati)**: clears at **₹4,450 / MWh**.
  - Collects **₹7.31 Crore** in transmission congestion surplus rent for CTU/NLDC grid expansion.
  - One-click restore button immediately resets the grid back to unconstrained unified clearing.
* **Regional Hub Clearing Inspector (`#acpRegionalHubModalOverlay`)**:
  - Interactive 5-tab switcher for deep inspection into state DISCOM physical drawals (Delhi BSES 4,120 MW, Punjab PSPCL 12,450 MW, Maharashtra MSEDCL 28,500 MW, Karnataka BESCOM 14,200 MW, West Bengal WBSEDCL 8,900 MW).
  - High-voltage corridor telemetry on 765kV AC and $\pm$800kV HVDC inter-ties (Wardha-Gwalior, Raigarh-Pugalur, Biswanath Chariyali-Agra).
  - Direct JSON dispatch schedule export.

---

## 📐 Mathematical Formulation & Deep Learning Theory

GridCast implements a mathematically grounded hybrid architecture combining recurrent memory dynamics, self-attention temporal weighting, and gradient-boosted residual correction:

### A. Gated Recurrent Unit (GRU) Temporal Dynamics
GRU addresses vanishing gradient degradation over long sequences through gating units:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \quad \text{// Update Gate}$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \quad \text{// Reset Gate}$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \quad \text{// Candidate Hidden State}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{// Final Hidden Representation}$$

### B. Self-Attention Alignment over 24-Hour Horizon
To dynamically weight extreme morning and evening ramps across the 24-step diurnal cycle:
$$\text{score}(\mathbf{h}_t, \mathbf{h}_i) = \mathbf{h}_t^T \mathbf{h}_i$$
$$\alpha_{t,i} = \frac{\exp(\text{score}(\mathbf{h}_t, \mathbf{h}_i))}{\sum_{j=1}^{24} \exp(\text{score}(\mathbf{h}_t, \mathbf{h}_j))}$$
$$\mathbf{c}_t = \sum_{i=1}^{24} \alpha_{t,i} \mathbf{h}_i$$

### C. Residual Gradient Boosting Coupling
The recurrent network generates baseline continuous predictions $\hat{y}_{\text{GRU}}(t)$. The non-linear residual errors are modeled using an Extreme Gradient Boosting (XGBoost) ensemble:
$$\hat{y}_{\text{final}}(t) = \hat{y}_{\text{GRU}}(t) + \hat{y}_{\text{XGBoost}}(\mathbf{X}_{\text{flattened}}(t))$$

* **Interactive Mathematical Formulation Modal (`#mathModalOverlay`)**: Directly accessible from the landing page, featuring interactive parameter tuning, step-by-step tensor visualizers, and simulation graphs.
* **Notebook Reference**: Full implementation and execution outputs are available in [Electricity_Demand_and_Price_forecasting_India.ipynb](http://localhost:5500/Electricity_Demand_and_Price_forecasting_India.ipynb) (cells 95–122) and the interactive web viewer [notebook-viewer.html](http://localhost:5500/notebook-viewer.html).

---

## 🛠️ Requirements & Installation

```bash
pip install pandas numpy scikit-learn statsmodels xgboost tensorflow matplotlib seaborn
```

To fetch or refresh live Indian weather data directly from Open-Meteo:
```bash
python3 fetch_indian_data.py --live --start-year 2000 --end-year 2025
```
