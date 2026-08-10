#!/bin/bash
# GRAND empirical pipeline (recommended hyperparameters).
# Edit project_path in config/settings.py before running.
# Hyperparameter search is time-consuming; the values below match one selected setting.

# Step 1: Prepare data
# -----------------------------------------------------------------------------
python empirical_analysis/valid_stocks.py
python empirical_analysis/feature.py
python empirical_analysis/labels.py

# -----------------------------------------------------------------------------
# Step 2: Train SGA mean / quantile models
# -----------------------------------------------------------------------------
# Mean model 
python empirical_analysis/train_quantile.py --tau 0.0 --mse-loss
# Quantile models
python empirical_analysis/train_quantile.py --tau 0.005
python empirical_analysis/train_quantile.py --tau 0.01
...
python empirical_analysis/train_quantile.py --tau 0.99
python empirical_analysis/train_quantile.py --tau 0.995

# -----------------------------------------------------------------------------
# Step 3: Quantile inference and QCM moments
# -----------------------------------------------------------------------------
python empirical_analysis/moment_inference.py

# -----------------------------------------------------------------------------
# Step 4: Features and labels for the variance (sigma) model
# -----------------------------------------------------------------------------
python empirical_analysis/sigma_feature.py
python empirical_analysis/sigma_label.py

# -----------------------------------------------------------------------------
# Step 5: Train variance model
# -----------------------------------------------------------------------------
python empirical_analysis/train_sigma.py

# -----------------------------------------------------------------------------
# Step 6: Train node-fusion MLPs
# -----------------------------------------------------------------------------
python empirical_analysis/train_mlp.py --tau 0.005
python empirical_analysis/train_mlp.py --tau 0.01
...
python empirical_analysis/train_mlp.py --tau 0.99
python empirical_analysis/train_mlp.py --tau 0.995

# -----------------------------------------------------------------------------
# Step 7: Estimate conditional covariance matrices
# -----------------------------------------------------------------------------
python empirical_analysis/estimate_covariance.py

# -----------------------------------------------------------------------------
# Step 8: Portfolio evaluation
# -----------------------------------------------------------------------------
python empirical_analysis/Markowitz_portfolio.py --method GRAND

# -----------------------------------------------------------------------------
# Step 9: Causal graph analysis
# -----------------------------------------------------------------------------
python empirical_analysis/extract_graph.py
python empirical_analysis/system_connectedness.py
python empirical_analysis/weekly_catfin.py
python empirical_analysis/spoilover_index.py
python empirical_analysis/bussiness_cycle.py

# Competing methods
python baselines/factor_model.py --method FF3
# ... other factor methods: FF5, FF6, CH4, HXZ4, PCA
python baselines/DCC_GARCH.py --mean FF3
# For the simplified GRAND method, run the following steps:
python baselines/nar_quantile.py --tau 0.0 --mse-loss
python baselines/nar_quantile.py --tau 0.005
python baselines/nar_quantile.py --tau 0.01
...
python baselines/nar_quantile.py --tau 0.99
python baselines/nar_quantile.py --tau 0.995
python baselines/nar_moment_inference.py
python baselines/nar_mlp.py --tau 0.005
python baselines/nar_mlp.py --tau 0.01
...
python baselines/nar_mlp.py --tau 0.99
python baselines/nar_mlp.py --tau 0.995
python baselines/nar_covariance.py

# -----------------------------------------------------------------------------
# Robustness checks
# -----------------------------------------------------------------------------
# Re-run related scripts with alternate hyperparameters or flags.
# Example (Table max-weight): vary --max in Markowitz_portfolio.py
#   python empirical_analysis/Markowitz_portfolio.py --method GRAND --max 0.01
# Example (sparse structure): vary --graph in estimate_covariance.py
#   python empirical_analysis/estimate_covariance.py --graph mean

# -----------------------------------------------------------------------------
# Global minimum variance portfolio (Appendix)
# -----------------------------------------------------------------------------
python empirical_analysis/GMV_portfolio.py --method ...

# -----------------------------------------------------------------------------
# Pairs trading (Appendix)
# -----------------------------------------------------------------------------
python empirical_analysis/pair_trading.py --method ...

# -----------------------------------------------------------------------------
# Simulation (Appendix)
# -----------------------------------------------------------------------------
# Use an isolated project_path for simulation artifacts.
# Simulated time indices are integers 0, 1, ..., T-1 (not calendar dates).
# Before Steps 2--7, set in config/settings.py e.g.
#   start_time = '0'; valid_time = '480'; end_time = '599'
python simulation/simulate_data.py
# After data generation, rerun Steps 2--7, replacing Step 4 with:
#   python simulation/sigma_data.py
