# README

**Manuscript title:** Machine Learning Vast Dynamic Conditional Covariance Matrices: the Spirit of "Divide and Conquer"

**Author(s):** Zhoufan Zhu, Ke Zhu

## 1. Overview

This repository reproduces the empirical and simulation results for the **GRAND** method. Set `project_path` in `config/settings.py`, place the manually downloaded raw data under that directory, then follow `run.sh` (Steps 1–9 for the main empirical pipeline; optional blocks for baselines, robustness, and simulation).

Recommended reading order:

1. Obtain licensed raw data and arrange directories as in Section 2.
2. Edit `config/settings.py` (`project_path` and sample windows).
3. Run the pipeline in Section 5 (or step through `run.sh`).



## 2. Data availability and provenance

This paper relies on commercial / subscription data. Raw inputs are **not** redistributed here; they must be downloaded manually from the providers below and saved under `{project_path}` with the names used by the code.


| Dataset                                                               | Source       | Sample / coverage                                       | Local path (under `{project_path}`)                                  |
| --------------------------------------------------------------------- | ------------ | ------------------------------------------------------- | -------------------------------------------------------------------- |
| Daily & weekly stock / index OHLCV                                    | Wind / CSMAR | A-shares, Shanghai & Shenzhen; 2010-01-01 to 2022-07-31 | `kline_day/`, `kline_week/`, `kline_day_index/`, `kline_week_index/` |
| Stock fundamentals & characteristics                                  | Wind / CSMAR | Same stocks and dates                                   | `basic_factor/`                                                      |
| Sector metadata (GICS)                                                | Wind / CSMAR | Static stock description                                | `overall_description.csv`                                            |
| Macro series (treasury bond, industrial production, social financing) | Wind / CSMAR | Weekly                                                  | `macro_data/macro_economy_week.csv`                                  |
| Risk-free / bond rates used in portfolio & cycle controls             | CSMAR / Wind | Daily bond yields                                       | `10Y_Bond.csv`, `6M_Bond.csv`                                        |
| China Macro-economic Climate Index (CMCI)                             | CEIC         | Monthly                                                 | `macro_data/CEIC_macro.csv`                                          |
| Chinese equity factors (monthly)                                      | PKU GSM      | Factor portfolios for baseline models                   | `factors_monthly_2023.xlsx`                                          |


**How to obtain (manual download).**

1. **Wind / CSMAR.** Through an institutional Wind or CSMAR terminal / API account, export for all A-share stocks listed on the Shanghai and Shenzhen exchanges over 2010-01-01–2022-07-31: daily and weekly prices/volumes, weekly fundamentals used as features, market-index klines, sector classification, and the bond / macro series listed above. Save one CSV per stock (or index) using the exchange ticker as the filename (e.g. `000001.XSHE.csv`, `000001.XSHG.csv` for the Shanghai Composite). Aggregate the three weekly macro series into `macro_data/macro_economy_week.csv`.
2. **CEIC.** From the CEIC database, download the monthly China Macro-economic Climate Index and save it as `macro_data/CEIC_macro.csv` with columns `date` (`mm/YYYY`) and `index`. This file is required only for the business-cycle analysis (`empirical_analysis/bussiness_cycle.py`).
3. **PKU factors.** Download the monthly Chinese equity factor file from [https://www.gsm.pku.edu.cn/finvc/info/1027/1147.htm](https://www.gsm.pku.edu.cn/finvc/info/1027/1147.htm) and save it as `factors_monthly_2023.xlsx`. Required for `baselines/factor_model.py` (and factor-demeaned DCC baselines).

After download, the raw tree should look like:

```text
{project_path}/
  kline_day/{stock}.csv
  kline_week/{stock}.csv
  kline_day_index/000001.XSHG.csv
  kline_week_index/000001.XSHG.csv
  basic_factor/{stock}.csv
  macro_data/macro_economy_week.csv
  macro_data/CEIC_macro.csv          # CEIC; business-cycle section only
  overall_description.csv
  10Y_Bond.csv
  6M_Bond.csv
  factors_monthly_2023.xlsx          # PKU factors; factor baselines only
```

All intermediate tensors, models, and results are written under the same `{project_path}` by the pipeline. Simulation artifacts should use a separate `project_path` (see `run.sh`).

**Sample design (as in the paper).** In-sample: 2010-01-01–2018-12-31 (80% train / 20% validation). Out-of-sample: 2019-01-01–2022-07-31. Stocks missing more than 20% of daily observations in-sample are dropped (`empirical_analysis/valid_stocks.py`), yielding N=1950 stocks, T=645 weeks, and P=31 features.

### Mocked data examples

Illustrative CSV snippets (not real observations). Column names and `date` formats must match; one file per stock/index under the kline and `basic_factor` folders.

`kline_day/000001.XSHE.csv` (daily stock):

```csv
date,open,high,low,close,volume,total_turnover,num_trades
2010-01-04,10.20,10.50,10.10,10.35,1200000,12420000,8500
2010-01-05,10.35,10.60,10.25,10.40,1100000,11440000,7900
```

`kline_week/000001.XSHE.csv` (weekly stock; same columns):

```csv
date,open,high,low,close,volume,total_turnover,num_trades
2010-01-08,10.20,10.80,10.05,10.55,6500000,68250000,42000
2010-01-15,10.55,10.90,10.40,10.70,6100000,65270000,40000
```

`kline_day_index/000001.XSHG.csv` and `kline_week_index/000001.XSHG.csv` (market index; same OHLCV fields):

```csv
date,open,high,low,close,volume,total_turnover,num_trades
2010-01-04,3243.75,3289.22,3230.15,3289.22,8.5e9,1.2e11,5200000
2010-01-05,3290.10,3310.00,3275.50,3295.40,7.9e9,1.1e11,4900000
```

`basic_factor/000001.XSHE.csv`:

```csv
date,a_share_market_val_in_circulation,du_return_on_equity_ttm,inc_revenue_ttm,total_asset_turnover_ttm,debt_to_asset_ratio_ttm
2010-01-08,1.2e11,0.15,0.08,0.45,0.62
2010-01-15,1.21e11,0.15,0.08,0.45,0.62
```

`overall_description.csv`:

```csv
order_book_id,sector_code
000001.XSHE,Financials
000002.XSHE,RealEstate
600000.XSHG,Financials
```

`macro_data/macro_economy_week.csv`:

```csv
date,treasury_bond,industrial,social_finance
2010-01-08,0.025,0.12,1.5e12
2010-01-15,0.025,0.11,1.52e12
```

`10Y_Bond.csv` / `6M_Bond.csv`:

```csv
date,10Y bond
2010-01-04,0.035
2010-01-05,0.0351
```

```csv
date,6M bond
2010-01-04,0.022
2010-01-05,0.0221
```

`macro_data/CEIC_macro.csv` (monthly CMCI; `date` as `mm/YYYY`):

```csv
date,index
01/2010,98.5
02/2010,99.1
```



## 3. Variable dictionaries



### Weekly stock kline (`kline_week/{stock}.csv`)


| Variable                       | Description        |
| ------------------------------ | ------------------ |
| `date`                         | Week date (index)  |
| `open`, `high`, `low`, `close` | Weekly OHLC prices |
| `volume`                       | Trading volume     |
| `total_turnover`               | Trading amount     |
| `num_trades`                   | Number of trades   |


Daily klines (`kline_day/`) and index klines use the same price/volume fields (index files omit stock-only fields as applicable).

### Stock fundamentals (`basic_factor/{stock}.csv`)


| Variable                            | Description                      |
| ----------------------------------- | -------------------------------- |
| `a_share_market_val_in_circulation` | Circulating A-share market value |
| `du_return_on_equity_ttm`           | Return on equity (TTM)           |
| `inc_revenue_ttm`                   | Revenue growth (TTM)             |
| `total_asset_turnover_ttm`          | Total asset turnover (TTM)       |
| `debt_to_asset_ratio_ttm`           | Debt-to-asset ratio (TTM)        |




### Macro (`macro_data/macro_economy_week.csv`)


| Variable         | Description                             |
| ---------------- | --------------------------------------- |
| `treasury_bond`  | One-year government bond rate           |
| `industrial`     | Growth rate of industrial production    |
| `social_finance` | Aggregate financing to the real economy |




### Other raw inputs


| File / variable                           | Description                                             |
| ----------------------------------------- | ------------------------------------------------------- |
| `overall_description.csv` → `sector_code` | GICS sector used for 11 sector dummies                  |
| `10Y_Bond.csv` → `10Y bond`               | 10-year bond yield (term-spread control)                |
| `6M_Bond.csv` → `6M bond`                 | 6-month bond yield (risk-free / relative-rate controls) |
| `CEIC_macro.csv` → `date`, `index`        | Monthly CMCI (`date` as `mm/YYYY`)                      |


Feature construction in `empirical_analysis/feature.py` builds the P=31 panel from market klines, fundamentals, macro series, and sector dummies (with train-window min-max normalization). Labels are horizon-ahead close-to-close returns (`empirical_analysis/labels.py`).

## 4. Computational requirements

- **OS / language:** Linux; Python 3.10
- **Core stack (as used in development):** PyTorch 2.5.1, CUDA 12.5
- **Main packages:** `numpy`, `pandas`, `torch`, `tensorboard`, `joblib`, `tqdm`, `cvxopt`, `statsmodels`, `scipy`
- **Baselines only:** `rpy2` plus an R environment with DCC-GARCH packages (`baselines/DCC_GARCH.py`)
- **Hardware:** a CUDA GPU is strongly recommended for SGA / MLP training; CPU is sufficient for data prep and most post-estimation scripts
- **Randomness:** training agents fix `seed=42` by default for reproducibility



### Runtime

This README includes run-time information because several scripts need more than a few minutes on a regular computer or laptop. Details below follow the computational-burden discussion in the supplementary materials (`draft/MS_supplement.tex`).

Due to deep learning models (e.g., SGA) and high-dimensional returns, GRAND is computationally heavy:

- **Simulation (per Monte Carlo sample).** On a single NVIDIA GeForce RTX 4090 GPU, forecasting one out-of-sample covariance matrix takes about **6 hours** on average. Most steps are parallelizable: with two RTX 4090 GPUs, wall-clock time falls by about **50%**. A machine with **32 GB** of memory is sufficient to reproduce the simulation study without parallelization.
- ++**Empirical pipeline (**++`run.sh` ++**Steps 2–7).** Training all quantile models, the variance model, node-fusion MLPs, and filling the sparse covariance matrices for N \approx 1950 stocks is on the order of **days** on one GPU; a full hyperparameter search is longer. Scripts default to the selected setting (++`hidden=128`++,++ `lr=0.0001`++,++ `lag=48`++). Relative to this vanilla implementation, an adaptive warm-start across nearby quantile levels cuts total trai++ning time by about **5%**, while jointly re-optimizing node-fusion parameters increases it by about **12%** (see the supplement).

Data preparation (Step 1) and most portfolio / graph post-processing scripts are much lighter and typically finish in minutes to a few hours on CPU, depending on N and I/O.

## 5. Programs / code

Edit `project_path` in `config/settings.py`, then run from the repository root. The master checklist is `run.sh`.

Training / inference scripts default to the selected hyperparameters (`--hidden 128`, `--lr 0.0001`, `--cuda 0`, `--lag` from `config/settings.py`), which write artifacts under the tag `hidden_128_lr_0.0001_lag_48_horizon_1`. Override via CLI flags when needed.

### Main empirical pipeline

```bash
# Step 1: Prepare data
python empirical_analysis/valid_stocks.py
python empirical_analysis/feature.py
python empirical_analysis/labels.py

# Step 2: Train SGA mean / quantile models
python empirical_analysis/train_quantile.py --tau 0.0 --mse-loss
python empirical_analysis/train_quantile.py --tau 0.005
# ... remaining tau in {0.01, ..., 0.99, 0.995}

# Step 3: Quantile inference and QCM moments
python empirical_analysis/moment_inference.py

# Step 4: Features and labels for the variance (sigma) model
python empirical_analysis/sigma_feature.py
python empirical_analysis/sigma_label.py

# Step 5: Train variance model
python empirical_analysis/train_sigma.py

# Step 6: Train node-fusion MLPs
python empirical_analysis/train_mlp.py --tau 0.005
# ... remaining tau in {0.01, ..., 0.99, 0.995}

# Step 7: Estimate conditional covariance matrices
python empirical_analysis/estimate_covariance.py

# Step 8: Portfolio evaluation
python empirical_analysis/Markowitz_portfolio.py --method GRAND

# Step 9: Causal graph / connectedness / business cycle
python empirical_analysis/extract_graph.py
python empirical_analysis/system_connectedness.py
python empirical_analysis/weekly_catfin.py
python empirical_analysis/spoilover_index.py
python empirical_analysis/bussiness_cycle.py
```


| Step | Produces (paper use)                                                                     |
| ---- | ---------------------------------------------------------------------------------------- |
| 1–7  | GRAND conditional covariance estimates                                                   |
| 8    | Markowitz portfolio tables                                                               |
| 9    | Causal graphs, system connectedness, CATFIN, spillover index, business-cycle regressions |




### Competing methods and simplified GRAND

See the “Competing methods” block in `run.sh` (`baselines/factor_model.py`, `baselines/DCC_GARCH.py`, and the `baselines/nar_*.py` sequence). Appendix scripts for GMV portfolios and pairs trading are also listed there.

### Robustness

Re-run portfolio / covariance scripts with alternate flags (examples in `run.sh`), e.g. `--max` in `Markowitz_portfolio.py` or `--graph` in `estimate_covariance.py`.

### Simulation (appendix)

Use a separate `project_path`. Simulated folders are **integer** time indices `0, 1, ..., T-1` (not calendar dates). Before Steps 2–7, set in `config/settings.py`, for example:

```python
start_time = '0'
valid_time = '480'
end_time = '599'
```

```bash
python simulation/simulate_data.py
# Then rerun Steps 2–7, replacing Step 4 with:
python simulation/sigma_data.py
```



### Layout


| Path                  | Role                                                         |
| --------------------- | ------------------------------------------------------------ |
| `config/`             | Shared paths and sample windows                              |
| `empirical_analysis/` | Data prep, training entry points, portfolios, graph analysis |
| `network/`            | SGA / MLP / QCM model and agents                             |
| `baselines/`          | Factor, DCC-GARCH, and simplified NAR variants               |
| `simulation/`         | Appendix data generation                                     |
| `run.sh`              | End-to-end reproduction checklist                            |




### Contact

Please feel free to raise an issue in this GitHub repository or email me ([tylerzzf@xmu.edu.com](mailto:tylerzzf@xmu.edu.com)) if you have any questions or encounter any issues.