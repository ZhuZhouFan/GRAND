"""
Compare GRAND dynamic connectedness with weekly CATFIN systemic-risk measures.

Inputs:
  - Pipeline: ``Q_graph.npy``, ``V_graph.npy``, ``overall_description.csv``,
    weekly klines, weekly index calendar.
  - External: ``{project_path}/10Y_Bond.csv`` (column ``10Y bond``; risk-free
    rate used to form excess financial-sector returns). Download / place this
    series under the data root before running.
Operations: estimate weekly CATFIN (GPD / SGED / nonparametric VaR average),
compute DGC from quantile and variance graphs, and plot OOS comparisons.
Outputs: ``{project_path}/macro_data/CATFIN.csv`` (if absent) and
``empirical_analysis/figures/weekly_DGCs_and_CATFIN.png``.
"""

import pandas as pd
import numpy as np
from scipy.stats import genpareto
from scipy.optimize import minimize, brentq
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('.')
from config import project_path, start_time, end_time

class SGED:
    def __init__(self):
        self.loc = None
        self.scale = None
        self.tail = None
        self.skew = None
        
    def pdf(self, sample:float, loc:float, scale:float, tail:float, skew:float) -> float:
        A = gamma_func(2 / tail) / np.sqrt(gamma_func(1 / tail)) / np.sqrt(gamma_func(3 / tail))
        S = np.sqrt(1 + (3 - 4 * A **2) * skew ** 2)
        theta = np.sqrt(gamma_func(1/tail)) / np.sqrt(gamma_func(3/tail)) / S
        C = tail / (2 * theta * gamma_func(1 / tail))
        delta = 2 * skew * A/S
        dominator = ((1 + np.sign(sample - loc + delta * scale) * skew)) ** tail * (theta ** tail) * (scale ** tail)
        numeritor = -1 * np.abs(sample - loc + delta * scale) ** tail
        density = C/scale * np.exp(numeritor / dominator)
        return density
    
    def fit(self, data):
        def object_func(params):
            loc, scale, tail, skew = params
            loglikelihoods = np.log(np.array([self.pdf(sample, loc, scale, tail, skew) for sample in data]) + 1e-6)
            return -np.sum(loglikelihoods)
            
        initial_params = [np.mean(data), np.std(data), 1.0, 0.0]
        result = minimize(object_func, initial_params, bounds=[(None, None), (1e-5, None), (1e-5, None), (-1, 1)])
        self.loc, self.scale, self.tail, self.skew = result.x
    
    def cdf(self, x: float) -> float:
        def integrand(t):
            return self.pdf(t, self.loc, self.scale, self.tail, self.skew)
        
        cdf_value, _ = quad(integrand, -np.inf, x)
        return cdf_value
    
    def quantile(self, alpha: float) -> float:
        def objective(x):
            return self.cdf(x) - alpha
        quantile_value = brentq(objective, -5, 5, maxiter  = 1000)
        return quantile_value
    
def calculate_weekly_CATFIN(fin_returns, selected_dates):
    
    VaRs = pd.DataFrame(columns=['GPD', 'SGED', 'NP'], index=selected_dates)
    VaRs.index.name = 'date'

    for date in tqdm(selected_dates[1:]):
        cs_data = fin_returns.loc[date, :].values
        bar = np.nanpercentile(cs_data, 0.1)
        ex_data = cs_data[cs_data <= bar]

        xi_hat, mu_hat, sigma_hat  = genpareto.fit(ex_data)
        VaRs.loc[date, 'GPD'] = genpareto.ppf(q = 0.01, c=xi_hat, loc = mu_hat, scale = sigma_hat)

        sged = SGED()
        sged.fit(ex_data)
        try:
            VaRs.loc[date, 'SGED'] = sged.quantile(0.01)
        except ValueError:
            pass
        
        VaRs.loc[date, 'NP'] = np.nanpercentile(cs_data, 0.01)

    return -1 * VaRs[['GPD', 'SGED', 'NP']].mean(axis = 1)

if __name__ == '__main__':
    
    des_df = pd.read_csv(f'{project_path}/overall_description.csv')
    fin_stocks = des_df.query("sector_code == 'Financials'")['order_book_id'].values
    index_week = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col = 'date')
    selected_dates = index_week.loc[start_time:end_time, :].index.values
    fin_returns = pd.DataFrame(columns=fin_stocks, index=selected_dates)

    for stock in fin_stocks:
        try:
            kline_week = pd.read_csv(f'{project_path}/kline_week/{stock}.csv', index_col='date')
        except FileNotFoundError:
            continue
        kline_week['return'] = np.log(kline_week['close']/kline_week['close'].shift(1))
        fin_returns[stock] = kline_week['return']

    bond_path = f'{project_path}/10Y_Bond.csv'
    if not os.path.exists(bond_path):
        raise FileNotFoundError(
            f'External risk-free series not found: {bond_path}. '
            f'Download / place 10Y_Bond.csv (column "10Y bond") under project_path.')
    bond_df = pd.read_csv(bond_path, index_col='date')
    fin_returns = fin_returns.dropna(axis=1, how='all')
    fin_returns = fin_returns.add(-1 * bond_df.loc[selected_dates, '10Y bond']/52 /100, axis=0)
    fin_returns.dropna(how='all', inplace = True)
    
    risk_df = pd.DataFrame(index=selected_dates, columns=['DGC_Q', 'DGC_V', 'CATFIN'])
    risk_df['CATFIN'] = calculate_weekly_CATFIN(fin_returns, selected_dates)

    os.makedirs(f'{project_path}/macro_data', exist_ok=True)
    if not os.path.exists(f'{project_path}/macro_data/CATFIN.csv'):
        risk_df['CATFIN'].to_csv(f'{project_path}/macro_data/CATFIN.csv')
    
    q_mat_dict = np.load(f'{project_path}/Q_graph.npy', allow_pickle=True).item()
    v_mat_dict = np.load(f'{project_path}/V_graph.npy', allow_pickle=True).item()
    tensor_dates = np.intersect1d(selected_dates, [x for x in q_mat_dict.keys()])
    for date in tqdm(tensor_dates):
        q_adj = q_mat_dict[date]
        v_adj = v_mat_dict[date]
        risk_df.loc[date, 'DGC_Q'] = q_adj.sum()/(q_adj.shape[0] * (q_adj.shape[0] - 1))
        risk_df.loc[date, 'DGC_V'] = v_adj.sum()/(v_adj.shape[0] * (v_adj.shape[0] - 1))
    
    # drop the in-sample part
    risk_df = risk_df.loc['2019-01-01':, :]
    risk_df.index = pd.to_datetime(risk_df.index)

    # visualization
    fig, (ax3, ax1) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(risk_df.index, 100 * risk_df['DGC_Q'], label=r'DGC$_{t,VaR}$', color='tab:blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('DGC(%)', color='black')

    ax2 = ax1.twinx() 
    ax2.plot(risk_df.index, risk_df['CATFIN'], label='CATFIN', color='tab:green', linestyle='--')
    ax2.set_ylabel('CATFIN', color='black')

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    ax3.plot(risk_df.index, 100 * risk_df['DGC_V'], label=r'DGC$_{t,\sigma}$', color='tab:red')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('DGC(%)', color='black')

    ax4 = ax3.twinx() 
    ax4.plot(risk_df.index, risk_df['CATFIN'], label='CATFIN', color='tab:green', linestyle='--')
    ax4.set_ylabel('CATFIN', color='black')
    
    lines = ax3.get_lines() + ax4.get_lines() 
    labels = [line.get_label() for line in lines]  
    ax3.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax3.transAxes)
    
    
    # highlight the important periods
    ax1.axvspan('2019-01-01', '2019-09-01', color='gray', alpha=0.3)  # (i) January 1, 2019 to September 1, 2019
    ax1.axvspan('2020-01-01', '2021-01-01', color='gray', alpha=0.3)  # (ii) January 1, 2020 to January 1, 2021
    ax1.axvspan('2021-11-01', '2022-06-01', color='gray', alpha=0.3)  # (iii) November 1, 2021 to June 1, 2022

    ax3.axvspan('2019-01-01', '2019-09-01', color='gray', alpha=0.3)
    ax3.axvspan('2020-01-01', '2021-01-01', color='gray', alpha=0.3)
    ax3.axvspan('2021-11-01', '2022-06-01', color='gray', alpha=0.3)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'weekly_DGCs_and_CATFIN.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {fig_path}')
    print('sample correlation matrix:', risk_df.corr())