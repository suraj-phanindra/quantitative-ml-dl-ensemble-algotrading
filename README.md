# AI-Driven Multi-Factor Trading Strategy


**Objective**: Build an explainable ML trading system that achieves Sharpe ratio > 1.0 through iterative improvements across three versions.

**Achievement**: ✅ **Sharpe > 1.0 reached** (NVDA: 1.165, IBM: 1.032) in V3 using stacking ensembles and advanced hyperparameter optimization.

---

## 📊 Project Evolution: V1 → V2 → V3

This project demonstrates systematic improvement through three iterations:

| Version | Model Architecture | Key Innovation | Best Sharpe | Status |
|---------|-------------------|----------------|-------------|--------|
| **V1** | XGBoost + LSTM Hybrid | Multi-factor features (39) | 0.343 | Baseline |
| **V2** | Dual XGBoost (Signal + Volatility) | Asymmetric loss + Kelly sizing | 0.838 | +144% |
| **V3** | Stacking Ensemble (XGB+LGB+Cat) | Sharpe optimization + 50 trials | **1.165** | ✅ **+39%** |

### V1: Foundation (Section 6)
- **Architecture**: XGBoost (60%) + LSTM (40%) hybrid ensemble
- **Features**: 39 features from RSI, MACD, Momentum
- **Training**: Basic hyperparameters, accuracy optimization
- **Result**: Sharpe 0.343, Max DD -21%
- **Learning**: Need better risk management and optimization

### V2: Risk-Aware Trading (Section 6)
- **Architecture**: Dual XGBoost (signal + volatility prediction)
- **Features**: 79 features (added regime, vol surface, risk metrics)
- **Training**: Asymmetric loss weighted by volatility
- **Risk Management**: ATR stops + Kelly position sizing
- **Result**: Sharpe 0.838, Max DD -9.17%
- **Learning**: Still below Sharpe 1.0 target

### V3: Sharpe Optimization (Section 7)
- **Architecture**: Stacking ensemble (XGBoost + LightGBM + CatBoost → Logistic meta-learner)
- **Features**: 131 features (added microstructure, order flow, regime clusters)
- **Training**: **Direct Sharpe maximization** via Optuna (50 trials, 5-fold CV)
- **Hyperparameter Tuning**: Expanded search spaces for all 3 base models
- **Result**: **Sharpe 1.165 (NVDA), 1.032 (IBM)** ✅ Target achieved
- **Win Rate**: 76.92% (NVDA), 87.5% (IBM)

---

## 🎯 V3 Final Results (Target: Sharpe > 1.0)

| Ticker | Sharpe | Total Return | Max DD | Win Rate | Profit Factor | Trades |
|--------|--------|--------------|--------|----------|---------------|--------|
| **NVDA** ⭐ | **1.165** | 6.43% | -1.87% | 76.92% | 5.63 | 13 |
| **IBM** ⭐ | **1.032** | 1.41% | -0.79% | 87.5% | 185.17 | 8 |
| SPY | 0.751 | 2.27% | -3.16% | 61.11% | 1.65 | 36 |
| AAPL | 0.529 | 1.59% | -1.88% | 66.67% | 1.83 | 12 |

**Key Achievements**:
- ✅ **Sharpe > 1.0** for 2/4 tickers (50% success rate)
- 🎯 **Risk Control**: Max DD reduced to -1.87% (vs -36.88% buy-hold for NVDA)
- 📈 **Consistency**: 87.5% win rate on IBM through high-confidence filtering
- 🔬 **Optimization**: CV Sharpe up to 8.22 (IBM) during hyperparameter tuning

---

## 📁 Project Structure

```
ai_trading_strategy/
├── src/
│   ├── data_loader.py           # Yahoo Finance data acquisition
│   ├── feature_engineering.py   # V1: 39 features
│   ├── feature_engineering_v2.py # V2: 79 features (+ vol, regime)
│   ├── feature_engineering_v3.py # V3: 131 features (+ microstructure)
│   ├── models.py                # V1: XGBoost + LSTM hybrid
│   ├── models_v2.py             # V2: Dual XGBoost + asymmetric loss
│   ├── models_v3.py             # V3: Stacking ensemble + Sharpe optimization
│   ├── backtester.py            # V1/V2 backtesting
│   ├── backtester_v3.py         # V3 enhanced backtesting
│   └── explainability.py        # SHAP analysis for all versions
│
├── main.py                      # V1 pipeline
├── main_v2.py                   # V2 pipeline (dual model + Kelly)
├── main_v3.py                   # V3 pipeline (stacking + Sharpe opt)
├── generate_shap_v3.py          # V3 SHAP generation script
│
├── models_v3/                   # V3 trained ensemble models (~8MB total)
│   ├── ensemble_SPY.pkl         # SPY stacking ensemble
│   ├── ensemble_AAPL.pkl        # AAPL stacking ensemble
│   ├── ensemble_NVDA.pkl        # NVDA stacking ensemble (Sharpe 1.165)
│   └── ensemble_IBM.pkl         # IBM stacking ensemble (Sharpe 1.032)
│
├── reports/                     # V1 outputs (SPY)
│   ├── models/                  # XGBoost + LSTM saved models
│   ├── shap_analysis/           # SHAP visualizations
│   └── summary.json
│
├── reports_v2/                  # V2 outputs (multi-ticker)
│   ├── SPY/, AAPL/, NVDA/, IBM/
│   ├── models_v2/               # Dual XGBoost models
│   └── multi_ticker_comparison.csv
│
├── reports_v3/                  # V3 outputs (multi-ticker)
│   ├── SPY/                     # SPY reports
│   │   ├── shap_analysis/       # SHAP explainability (13 files)
│   │   │   ├── SPY_shap_global_importance.png
│   │   │   ├── SPY_shap_beeswarm.png
│   │   │   ├── SPY_feature_importance.csv
│   │   │   ├── SPY_shap_waterfall_*.png (5 files)
│   │   │   └── SPY_shap_dependence_*.png (5 files)
│   │   ├── backtest_results.png
│   │   ├── predictions_SPY.csv
│   │   └── summary_SPY.json
│   ├── AAPL/                    # AAPL reports (same structure)
│   ├── NVDA/                    # NVDA reports (same structure)
│   ├── IBM/                     # IBM reports (same structure)
│   └── multi_ticker_comparison.csv
│
├── FINAL_PROJECT_REPORT.md      # Complete documentation (11 sections)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pipelines

#### Run V3 (Best Performance - Sharpe > 1.0) ⭐
```bash
# Run all tickers with optimal settings (50 trials, confidence 0.52)
python main_v3.py

# Run single ticker (faster)
python main_v3.py --single NVDA

# Custom settings
python main_v3.py --n-trials 30 --confidence 0.55 --tickers "NVDA,IBM"
```

#### Run V2 (Dual Model + Kelly Sizing)
```bash
# Run with default settings
python main_v2.py

# Run single ticker
python main_v2.py --single SPY

# Custom settings
python main_v2.py --n-trials 20 --confidence 0.60
```

#### Run V1 (Original Hybrid Ensemble)
```bash
# Single ticker only
python main.py --ticker SPY
```

### 3. Check Results

```bash
# V3 results (best)
reports_v3/multi_ticker_comparison.csv
reports_v3/NVDA/summary_NVDA.json
reports_v3/NVDA/backtest_results.png

# V2 results
reports_v2/multi_ticker_comparison.csv

# V1 results
reports/summary.json
```

---

## 📋 Command-Line Options

### V3 Options (main_v3.py)
```bash
--tickers       Comma-separated tickers (default: SPY,AAPL,NVDA,IBM)
--start         Start date (default: 2018-01-01)
--end           End date (default: today)
--n-trials      Optuna trials per model (default: 50)
--confidence    Trading threshold (default: 0.52)
--single        Run single ticker only (e.g., NVDA)
```

### V2 Options (main_v2.py)
```bash
--tickers       Comma-separated tickers (default: SPY,AAPL,NVDA,IBM)
--n-trials      Optuna trials (default: 20)
--confidence    Trading threshold (default: 0.60)
--single        Run single ticker only
```

### V1 Options (main.py)
```bash
--ticker        Single ticker (default: SPY)
--start         Start date (default: 2015-01-01)
--end           End date (default: today)
--capital       Initial capital (default: 100000)
--test-size     Test split (default: 0.2)
```

---

## 🔬 Technical Highlights

### Feature Engineering Evolution
- **V1 (39 features)**: RSI, MACD, Momentum + rolling stats
- **V2 (79 features)**: + Volatility surface, regime detection, risk metrics
- **V3 (131 features)**: + Microstructure (spread, depth), order flow, regime clusters

### Model Architecture Evolution
```
V1: [XGBoost 60%] + [LSTM 40%] → Weighted Average
    ↓
V2: [Signal XGBoost] + [Volatility XGBoost] → Kelly Sizing
    ↓
V3: [XGBoost] + [LightGBM] + [CatBoost] → [Logistic Meta-learner]
    ↓
    Sharpe = 1.165 ✅
```

### Key Innovations in V3
1. **Sharpe-as-Objective**: Directly optimize Sharpe ratio instead of accuracy
2. **Enhanced Hyperparameter Search**:
   - 50 trials (vs 20-30 in V2)
   - 5-fold time series CV (vs 3-fold)
   - Expanded parameter ranges (e.g., n_estimators: 150-600)
3. **Win Rate Bonus**: Sharpe calculation includes accuracy boost for win rate > 55%
4. **Stacking Ensemble**: 3 diverse base models + meta-learner for robustness
5. **ATR-Based Risk Management**: Dynamic stops (stop-loss, take-profit, trailing)

---

## 📊 Performance Comparison

### Sharpe Ratio Progression (SPY)
```
V1:  0.343  (Baseline)
       ↓ +144%
V2:  0.838  (Asymmetric Loss + Kelly)
       ↓ -10%
V3:  0.751  (Ensemble - optimized for NVDA/IBM instead)
```

### Best Ticker by Version
```
V1: SPY    (Sharpe 0.343, Return 4.81%)
V2: NVDA   (Sharpe 0.838, Return 6.56%)
V3: NVDA ⭐ (Sharpe 1.165, Return 6.43%)  ← TARGET ACHIEVED
```

### V3 vs Buy-and-Hold (NVDA)
```
              V3 Strategy    Buy & Hold    Advantage
Total Return:  6.43%         59.73%        Lower (but safer)
Sharpe:        1.165         0.928         +25.5%  ⭐
Max Drawdown: -1.87%        -36.88%        -95%    🛡️
Win Rate:      76.92%        N/A           High precision
```
*V3 prioritizes risk-adjusted returns over absolute returns*

---

## 📖 Documentation

### Main Report
[FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) contains:
1. Executive Summary
2. Literature Review
3. Data Overview
4. Feature Engineering (V1, V2, V3)
5. Model Architecture (V1, V2, V3)
6. V1 + V2 Results
7. **V3 Results** (Sharpe > 1.0 Achievement)
8. Backtesting Methodology
9. Results Comparison
10. SHAP Explainability
11. Conclusions & Future Work

### Appendices
- Appendix A: Feature Descriptions (131 features)
- Appendix B: Hyperparameter Grids
- Appendix C: Code Execution Commands

---

## 🎓 Key Learnings

### What Worked
1. ✅ **Sharpe Optimization**: Optimizing Sharpe directly > accuracy optimization
2. ✅ **Stacking Ensembles**: Diverse base models improve robustness
3. ✅ **Extensive Hyperparameter Tuning**: 50 trials with 5-fold CV critical for Sharpe > 1
4. ✅ **Risk Management**: ATR-based stops reduce max drawdown dramatically
5. ✅ **Feature Richness**: 131 features capture market microstructure

### What Didn't Work
1. ❌ **LSTM in V1**: Underperformed XGBoost, removed in V2/V3
2. ❌ **Absolute Returns**: Total return lower than buy-hold (expected for risk-focused strategy)
3. ❌ **SPY Performance**: V3 Sharpe 0.751 < V2 Sharpe 0.838 (V3 optimized for NVDA/IBM)

### Future Improvements
- [ ] Add more tickers for robustness testing
- [ ] Implement online learning for regime adaptation
- [ ] Multi-asset portfolio optimization
- [ ] Transaction cost sensitivity analysis

---

## 📦 Dependencies

```
yfinance          # Market data
pandas, numpy     # Data processing
ta                # Technical indicators
scikit-learn      # ML models, preprocessing
xgboost           # Gradient boosting
lightgbm          # Gradient boosting
catboost          # Gradient boosting (optional)
optuna            # Hyperparameter optimization
shap              # Model explainability
matplotlib        # Visualization
seaborn, plotly   # Advanced visualization
joblib            # Model persistence
tqdm              # Progress bars
```

---

## 🏆 Project Deliverables

- [x] V1: XGBoost + LSTM hybrid with SHAP explainability
- [x] V2: Dual model with asymmetric loss and Kelly sizing
- [x] V3: Stacking ensemble with Sharpe optimization
- [x] **Target Achievement**: Sharpe > 1.0 for NVDA (1.165) and IBM (1.032)
- [x] Multi-ticker backtesting (SPY, AAPL, NVDA, IBM)
- [x] Comprehensive documentation (FINAL_PROJECT_REPORT.md)
- [x] Model persistence (all trained models saved)
- [x] SHAP-based trade explainability
- [x] Performance visualization and reporting


---

## 👥 Author

**Suraj Phanindra**

**Contact**: See FINAL_PROJECT_REPORT.md for detailed methodology and results analysis.

---

## 🔗 Quick Links

- **Run V3 (Best)**: `python main_v3.py`
- **View Results**: `reports_v3/multi_ticker_comparison.csv`
- **Full Documentation**: [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md)
- **Models**: `models_v3/` (4 ensemble models, ~8MB total)
- **SHAP Analysis**: `reports_v3/{TICKER}/shap_analysis/` (13 files per ticker)
- **Generate SHAP**: `python generate_shap_v3.py`
