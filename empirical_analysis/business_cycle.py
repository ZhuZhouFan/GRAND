"""
Business-cycle predictive regressions for GRAND connectedness measures.

Inputs:
  - Pipeline: ``Q_graph.npy``, ``V_graph.npy`` from ``extract_graph.py``;
    ``spillover_index_{tau}.csv`` from ``spillover_index.py``; stock /
    sector / basic-factor panels already used upstream.
  - External (place under ``{project_path}`` before running):
      * ``macro_data/CEIC_macro.csv`` — monthly macro index (``date`` as
        ``mm/YYYY``, column ``index``)
      * ``10Y_Bond.csv`` — column ``10Y bond``
      * ``6M_Bond.csv`` — column ``6M bond``
      * ``kline_day_index/000001.XSHG.csv`` — daily market calendar / prices
Operations: build monthly DGC / spillover predictors and controls (CATFIN,
term spread, relative rate, market and financial moments, CEIC leads/lags),
run Newey-West OLS for horizons 1..12, and plot coefficient paths.
Outputs: printed R^2 / CIs and
``empirical_analysis/figures/business_cycle_analysis.png``.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import genpareto

import sys
sys.path.append('.')
from config import project_path, valid_time, end_time
from empirical_analysis.weekly_catfin import SGED

def load_spillover_index(project_path, tau=0.05):
    spillover_df = pd.read_csv(f'{project_path}/macro_data/spillover_index_{tau}.csv', index_col='date')
    spillover_df.index = pd.to_datetime(spillover_df.index, format='%Y-%m-%d')
    spillover_df = spillover_df.resample('ME').last()
    return spillover_df.rename(columns={'index': 'SPILLOVER_INDEX'})

def calculate_TREM(project_path):
    bond10y_df = pd.read_csv(f'{project_path}/10Y_Bond.csv', index_col = 'date')
    bond10y_df.index = pd.to_datetime(bond10y_df.index, format='%Y-%m-%d')
    bond6m_df = pd.read_csv(f'{project_path}/6M_Bond.csv', index_col = 'date')
    bond6m_df.index = pd.to_datetime(bond6m_df.index, format='%Y-%m-%d')

    trem_df = bond10y_df['10Y bond'] - bond6m_df['6M bond']
    trem_df = trem_df.resample('ME').last()
    return trem_df

def calculate_RREL(project_path):
    bond6m_df = pd.read_csv(f'{project_path}/6M_Bond.csv', index_col = 'date')
    bond6m_df.index = pd.to_datetime(bond6m_df.index, format='%Y-%m-%d')
    
    rrel_df = bond6m_df['6M bond'] - bond6m_df['6M bond'].rolling(window=252).mean()
    rrel_df = rrel_df.resample('ME').last()
    return rrel_df

def calculate_market_excess_return(project_path):
    index_day = pd.read_csv(f'{project_path}/kline_day_index/000001.XSHG.csv', index_col = 'date')
    index_day.index = pd.to_datetime(index_day.index, format='%Y-%m-%d')
    monthly_prices = index_day.resample('ME').last()
    
    bond10y_df = pd.read_csv(f'{project_path}/10Y_Bond.csv', index_col = 'date')
    bond10y_df.index = pd.to_datetime(bond10y_df.index, format='%Y-%m-%d')
    
    monthly_returns = monthly_prices['close'].pct_change(fill_method=None) - bond10y_df['10Y bond'].resample('ME').last()/52/100
    return monthly_returns
    
def calculate_market_volatility(project_path):
    index_day = pd.read_csv(f'{project_path}/kline_day_index/000001.XSHG.csv', index_col = 'date')
    index_day.index = pd.to_datetime(index_day.index, format='%Y-%m-%d')
    index_day['return'] = index_day['close'].pct_change(fill_method=None)
    
    return index_day['return'].resample('ME').std()

def calculate_financials_covariates(project_path):
    bond10y_df = pd.read_csv(f'{project_path}/10Y_Bond.csv', index_col = 'date')
    bond10y_df.index = pd.to_datetime(bond10y_df.index, format='%Y-%m-%d')
    index_day = pd.read_csv(f'{project_path}/kline_day_index/000001.XSHG.csv', index_col = 'date')
    index_day.index = pd.to_datetime(index_day.index, format='%Y-%m-%d')
    
    des_df = pd.read_csv(f'{project_path}/overall_description.csv')
    fin_stocks = des_df.query("sector_code == 'Financials'")['order_book_id'].values
    fin_prices = pd.DataFrame(columns=fin_stocks, index=index_day.index)
    fin_cap = pd.DataFrame(columns=fin_stocks, index=index_day.index)
    
    for stock in fin_stocks:
        try:
            kline_day = pd.read_csv(f'{project_path}/kline_day/{stock}.csv', index_col='date')
            kline_day.index = pd.to_datetime(kline_day.index, format='%Y-%m-%d')
            fin_prices[stock] = kline_day['close']
        except FileNotFoundError:
            continue
        
        try:
            basic_df = pd.read_csv(f'{project_path}/basic_factor/{stock}.csv', index_col='date')
            basic_df.index = pd.to_datetime(basic_df.index, format='%Y-%m-%d')
            fin_cap[stock] = basic_df['a_share_market_val_in_circulation']
        except FileNotFoundError:
            continue
    
    fin_monthly_prices = fin_prices.resample('ME').last()
    fin_monthly_returns = fin_monthly_prices.pct_change(fill_method=None).add(-1 * bond10y_df['10Y bond'].resample('ME').last()/52/100, axis = 0).astype(np.float64)
    fin_monthly_cap = fin_cap.resample('ME').last()
    
    fin_vwa_ret = (fin_monthly_returns * fin_monthly_cap.mul(1/fin_monthly_cap.sum(axis = 1), axis = 0)).sum(axis = 1).astype(np.float64)
    fin_vwa_vol = (fin_prices.pct_change(fill_method=None).resample('ME').std() * fin_monthly_cap.mul(1/fin_monthly_cap.sum(axis = 1), axis = 0)).sum(axis = 1).astype(np.float64)
    fin_vwa_skew = (fin_prices.pct_change(fill_method=None).resample('ME').apply(lambda x: x.skew()) * fin_monthly_cap.mul(1/fin_monthly_cap.sum(axis = 1), axis = 0)).sum(axis = 1).astype(np.float64)
    
    return fin_vwa_ret, fin_vwa_vol, fin_vwa_skew, fin_monthly_returns

def calculate_monthly_CATFIN(fin_monthly_returns):
    var_df = pd.DataFrame(index=fin_monthly_returns.index, columns=['GPD', 'SGED', 'NP'], dtype=np.float64)
    for date in fin_monthly_returns.index.values:
        cs_data = fin_monthly_returns.loc[date, :].values

        if fin_monthly_returns.shape[1] - np.isnan(cs_data).sum() < 30:
            continue
        
        bar = np.nanpercentile(cs_data, 0.1)
        ex_data = cs_data[cs_data <= bar]

        xi_hat, mu_hat, sigma_hat  = genpareto.fit(ex_data)
        var_df.loc[date, 'GPD'] = genpareto.ppf(q = 0.01, c=xi_hat, loc = mu_hat, scale = sigma_hat)

        sged = SGED()
        sged.fit(ex_data)
        try:
            var_df.loc[date, 'SGED'] = sged.quantile(0.01)
        except ValueError:
            pass
        
        var_df.loc[date, 'NP'] = np.nanpercentile(cs_data, 0.01)
        
    return -1 * var_df[['GPD', 'SGED', 'NP']].mean(axis = 1)

if __name__ == "__main__":
    required_external = [
        f'{project_path}/macro_data/CEIC_macro.csv',
        f'{project_path}/10Y_Bond.csv',
        f'{project_path}/6M_Bond.csv',
        f'{project_path}/kline_day_index/000001.XSHG.csv',
        f'{project_path}/Q_graph.npy',
        f'{project_path}/V_graph.npy',
        f'{project_path}/macro_data/spillover_index_0.05.csv',
    ]
    missing = [p for p in required_external if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Missing required inputs for business-cycle analysis:\n  - '
            + '\n  - '.join(missing))

    # load CEIC data
    ceic_index = pd.read_csv(f'{project_path}/macro_data/CEIC_macro.csv')
    # set the index to be the last day of the month
    ceic_index['date'] = pd.to_datetime(ceic_index['date'], format='%m/%Y') + pd.offsets.MonthEnd(0)
    ceic_index.set_index('date', inplace=True)

    index_week = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    selected_dates = index_week.index.values

    # process the DGCs (filenames match extract_graph.py outputs)
    dgc_df = pd.DataFrame(index=selected_dates, columns=['DGC_Q', 'DGC_V'], dtype=np.float64)
    q_mat_dict = np.load(f'{project_path}/Q_graph.npy', allow_pickle=True).item()
    v_mat_dict = np.load(f'{project_path}/V_graph.npy', allow_pickle=True).item()
    tensor_dates = np.intersect1d(selected_dates, [x for x in q_mat_dict.keys()])
    for date in tensor_dates:
        q_adj = q_mat_dict[date]
        v_adj = v_mat_dict[date]
        dgc_df.loc[date, 'DGC_Q'] = q_adj.sum() / (q_adj.shape[0] * (q_adj.shape[0] - 1))
        dgc_df.loc[date, 'DGC_V'] = v_adj.sum() / (v_adj.shape[0] * (v_adj.shape[0] - 1))

    # resample the DGC_df to monthly frequency and take the mean
    dgc_df.index = pd.to_datetime(dgc_df.index, format='%Y-%m-%d')
    dgc_df = dgc_df.resample('ME').mean()

    # load spillover index data
    spillover_df = load_spillover_index(project_path, tau=0.05)
    
    # combine all indices
    all_indices_df = pd.concat([dgc_df, spillover_df], axis=1)
    
    # process the control variables
    control_df = pd.DataFrame(index = ceic_index.index,
                              columns = ['TREM', 'RREL', 'MKT_RET', 'MKT_VOL', 'FIN_RET', 'FIN_VOL', 'FIN_SKEW']
                              + [f'Macro_lag_{x}' for x in range(12)] 
                              + [f'Macro_{x}' for x in range(1, 13)],
                              dtype=np.float64)
    
    # calculate the TREM
    control_df['TREM'] = calculate_TREM(project_path)
    
    # calculate the RREL
    control_df['RREL'] = calculate_RREL(project_path)
    
    # calculate the market excess return
    control_df['MKT_RET'] = calculate_market_excess_return(project_path)
    
    # calculate the market volatility
    control_df['MKT_VOL'] = calculate_market_volatility(project_path)
    
    # calculate the financials covariates
    control_df['FIN_RET'], control_df['FIN_VOL'], control_df['FIN_SKEW'], fin_monthly_returns = calculate_financials_covariates(project_path)
    
    # calculate the monthly CATFIN
    control_df['CATFIN'] = calculate_monthly_CATFIN(fin_monthly_returns)
    
    # construct the leading and lagging ceic variables
    for lead in range(1, 13):
        control_df[f'Macro_{lead}'] = ceic_index['index'].shift(-lead)
    
    for lag in range(12):
        control_df[f'Macro_lag_{lag}'] = ceic_index['index'].shift(lag)
    
    # process the dataset for regression analysis
    data_df = pd.concat([all_indices_df, control_df], axis=1)
    data_df.dropna(how = 'any', inplace= True)
    data_df = data_df.loc[valid_time:end_time, :]
    
    # sample size and dimensions of the dataset
    print(f"Dataset shape: {data_df.shape}")
    
    estimations = {}
    label_type = 'Macro'
    
    index_types = ['DGC_V', 'DGC_Q', 'SPILLOVER_INDEX']
    index_labels = ['DGC_V', 'DGC_Q', 'SPILLOVER']
    
    for idx, index_type in enumerate(index_types):
        index_label = index_labels[idx]
        estimations[index_label] = {}

        X = data_df[['CATFIN'] + [f'Macro_lag_{x}' for x in range(12)] 
                    + ['TREM', 'RREL', 'MKT_RET', 'MKT_VOL', 'FIN_RET', 'FIN_VOL', 'FIN_SKEW']
                    + [index_type]].values
        X = sm.add_constant(X)

        CIs = np.zeros([12, 2], dtype=np.float64)
        betas = np.zeros([12], dtype=np.float64)
        R2s = np.zeros([12], dtype=np.float64)

        for horizon in range(1, 13):
            y = data_df[f'{label_type}_{horizon}'].values/100
            model = sm.OLS(y, X)
            results = model.fit()
            newey_west_se = results.get_robustcov_results(cov_type='HAC', maxlags=5)
            params = newey_west_se.params
            conf_int = newey_west_se.conf_int(alpha=0.05)
            betas[horizon - 1] = params[-1]
            CIs[horizon - 1, :] = conf_int[-1, :]
            R2s[horizon - 1] = results.rsquared_adj

        estimations[index_label]['beta'] = betas
        estimations[index_label]['CI'] = CIs
        estimations[index_label]['R2'] = R2s
    
    # print the R squares
    for index_label in index_labels:
        print(f'{index_label} R^2 for different horizons', estimations[index_label]['R2'])
    
    # visualize the confs and coefs with three subplots side by side
    lags = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    label_fontsize = 20
    tick_fontsize = 16
    subplot_label_fontsize = 20

    for idx, index_label in enumerate(index_labels):
        ax = axes[idx]
        ax.plot(lags, estimations[index_label]['beta'],
                color='black', linewidth=2.5)
        ax.plot(lags, estimations[index_label]['CI'][:, 0],
                linestyle='--', color='black', linewidth=1.8)
        ax.plot(lags, estimations[index_label]['CI'][:, 1],
                linestyle='--', color='black', linewidth=1.8)
        ax.axhline(0, color='gray', linewidth=1.0)

        print(f'{index_label} CI:\n', estimations[index_label]['CI'])

        ax.set_xlabel("Horizon", fontsize=label_fontsize)
        ax.set_xticks(ticks=lags, labels=[f"{i}" for i in lags])
        # ax.set_xticks(ticks=[3, 6, 9, 12], labels=['3', '6', '9', '12'])
        ax.tick_params(axis='x', which='major', labelsize=tick_fontsize + 4)
        ax.tick_params(axis='y', which='major', labelsize=tick_fontsize)

        if index_label == 'DGC_Q':
            ax.set_ylabel(r"Coefficient of $\mathrm{DGC}_{t,\mathrm{VaR}}$",
                          fontsize=label_fontsize)
        elif index_label == 'DGC_V':
            ax.set_ylabel(r"Coefficient of $\mathrm{DGC}_{t,\sigma}$",
                          fontsize=label_fontsize)
        else:
            ax.set_ylabel(r"Coefficient of $\mathrm{SPILLOVER}_{t}$",
                          fontsize=label_fontsize)

        subplot_label = chr(ord('a') + idx)
        ax.text(0.5, -0.28, f"({subplot_label})",
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=subplot_label_fontsize)

    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    output_filename = os.path.join(fig_dir, 'business_cycle_analysis.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Analysis complete. Results saved to {output_filename}')