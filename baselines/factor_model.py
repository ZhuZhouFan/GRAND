import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.decomposition import PCA

import sys
sys.path.append('.')
from config import project_path, start_time, end_time


def load_factors(selected_dates, factor_list):
    monthly_factor_df = pd.read_excel(f'{project_path}/factors_monthly_2023.xlsx', index_col='trdmnt')
    weekly_factor_df = pd.DataFrame(
        index=selected_dates, columns=factor_list, dtype=float)
    weekly_factor_df.loc[:, :] = 0.0

    for date_str in selected_dates:
        date_obj = pd.to_datetime(date_str)
        target_month = date_obj.strftime('%Y-%m')

        weekly_factor_df.loc[date_str, factor_list] = monthly_factor_df.loc[target_month,
                                                                            factor_list].values.astype(float)

    return weekly_factor_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)
    index_week = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    selected_dates = index_week.loc[start_time:end_time, :].index.values
    est_dates = selected_dates[50:] # at least 50 samples are kept for rolling estimation
    bond_df = pd.read_csv(f'{project_path}/10Y_Bond.csv', index_col='date')
    ret_matrix = pd.DataFrame(index=selected_dates, columns=stock_list)
    for stock in stock_list:
        try:
            kline_week = pd.read_csv(
                f'{project_path}/kline_week/{stock}.csv', index_col='date')
        except FileNotFoundError:
            continue
        kline_week['return'] = kline_week['close']/kline_week['close'].shift(1) - 1
        ret_matrix[stock] = kline_week['return']
    aver_risk_free = bond_df.loc[start_time:end_time, '10Y bond'].mean()/100
    
    ret_matrix_nna = ret_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
    ret_matrix_nna = ret_matrix_nna.iloc[1:, :] # remove the all zeros in 1st row.

    save_path = f'{project_path}/{args.method}'
    os.makedirs(save_path, exist_ok=True)
    
    if args.method == 'FF3':
        factor_list = ['mkt', 'smbff3', 'hmlff3']
        factor_df = load_factors(selected_dates, factor_list)
    elif args.method == 'FF5':
        factor_list = ['mkt', 'smbff5', 'hmlff5', 'cma', 'rmw']
        factor_df = load_factors(selected_dates, factor_list)
    elif args.method == 'FF6':
        factor_list = ['mkt', 'smbff5', 'hmlff5', 'cma', 'rmw', 'mom']
        factor_df = load_factors(selected_dates, factor_list)
    elif args.method == 'CH4':
        factor_list = ['mkt', 'smbff3', 'hmlff3', 'mom']
        factor_df = load_factors(selected_dates, factor_list)
    elif args.method == 'HXZ4':
        factor_list = ['mkt', 'me', 'ia', 'roe']
        factor_df = load_factors(selected_dates, factor_list) 
    elif args.method == 'PCA':
        pass
    else:
        raise ValueError
    
    for date in tqdm(est_dates):
        
        prev_date = (pd.to_datetime(date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        window_rets = ret_matrix_nna.loc[prev_date:date, :] - aver_risk_free
        
        if args.method == 'PCA':
            best_ic = np.inf
            T, N = window_rets.shape
            
            for factor_num in range(1, 10):
                pca = PCA(n_components=factor_num)
                pca_factors = pca.fit_transform(window_rets)
                pca_factors = pd.DataFrame(pca_factors, index=window_rets.index)

                X = pca_factors.T @ pca_factors
                Y = pca_factors.T @ window_rets
                Gamma = np.linalg.solve(X, Y)

                residuals = window_rets - (pca_factors @ Gamma).values

                ic_value = np.log(np.mean(residuals ** 2)) + factor_num * (N + T)/(N * T) * np.log(N * T/(N + T))
                
                if ic_value < best_ic:
                    best_pca_factors = pca_factors
                    best_Gamma = Gamma
                    best_residual = residuals

                    best_ic = ic_value
            
            pred_f = best_pca_factors.rolling(10).mean().loc[date, :].values
            pred_mu = pred_f @ best_Gamma + aver_risk_free

            cov_f = best_pca_factors.cov()
            cov_u = best_residual.cov()
            pred_sigma = (best_Gamma.T @ cov_f @ best_Gamma).values + cov_u.values
        else:
            window_factors = factor_df.loc[window_rets.index, :]

            X = window_factors.T @ window_factors
            Y = window_factors.T @ window_rets
            Gamma = np.linalg.solve(X, Y)

            pred_f = factor_df.rolling(10).mean().loc[date, :].values
            pred_mu = pred_f @ Gamma + aver_risk_free

            cov_f = window_factors.cov()
            residuals = window_rets - (window_factors @ Gamma).values
            cov_u = residuals.cov()

            pred_sigma = (Gamma.T @ cov_f @ Gamma).values + cov_u.values

        np.save(f'{save_path}/{date}_mean', pred_mu)
        np.save(f'{save_path}/{date}_cov', pred_sigma)
