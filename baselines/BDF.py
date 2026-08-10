import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
import numpy as np
import pandas as pd
import os
import argparse

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, end_time

pandas2ri.activate()
numpy2ri.activate()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=str, default='FF3', help='How to demean')
    parser.add_argument('--num_factor', type=int, default=4, help='Number of latent factors')
    args = parser.parse_args()
    
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)
    index_week = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    selected_dates = index_week.loc[start_time:end_time, :].index.values
    ret_matrix = pd.DataFrame(index=selected_dates, columns=stock_list)
    
    for stock in stock_list:
        try:
            kline_week = pd.read_csv(f'{project_path}/kline_week/{stock}.csv', index_col='date')
        except FileNotFoundError:
            continue
        kline_week['return'] = kline_week['close']/kline_week['close'].shift(1) - 1
        ret_matrix[stock] = kline_week['return']
    
    save_path = f'{project_path}/{args.mean}_BDF'
    os.makedirs(save_path, exist_ok=True)
        
    if args.mean in ['FF3', 'FF5', 'PCA']:
        mean_matrix = pd.DataFrame(columns=stock_list, index = selected_dates, dtype=float)
        for date in selected_dates:
            try:
                pred_mean = np.load(f'{project_path}/{args.mean}/{date}_mean.npy')
                mean_matrix.loc[date, :] = pred_mean.astype(float)
            except FileNotFoundError:
                continue
        
    elif args.mean == 'GRAND':
        mean_matrix = pd.DataFrame(columns=stock_list, index = selected_dates, dtype=float)
        for stock in stock_list:
            try:
                moment_df = pd.read_csv(f'{project_path}/moment/hidden_128_lr_0.0001_lag_48_horizon_1/{stock}.csv', index_col='date')
            except FileNotFoundError:
                continue
            mean_matrix[stock] = moment_df['mean']
    else:
        raise ValueError('Misspecification in Mean model')
        
    demean_ret_matrix = ret_matrix - mean_matrix
    demean_ret_matrix.dropna(how='all', inplace = True)
    demean_ret_matrix = demean_ret_matrix.apply(lambda row: row.fillna(row.median()), axis=1)
        
    T, N = demean_ret_matrix.shape
        
    in_sample_data = demean_ret_matrix.loc[start_time:valid_time, :]
    train_size = in_sample_data.shape[0]
    oos_dates = demean_ret_matrix.index[train_size:]
    H = len(oos_dates)

    importr('factorstochvol')
    
    robjects.globalenv['returns'] = in_sample_data
    robjects.globalenv['num_factor'] = args.num_factor
    robjects.globalenv['N'] = N
    robjects.globalenv['H'] = H
    robjects.r('''
               
    library(factorstochvol)
    set.seed(42)
    Rtn <- t(returns)
    
    # estimate the model
    res = fsvsample(Rtn, factors = num_factor, draws = 100, burnin = 10)
    
    pred_covs <- array(0, dim = c(N, N, H))
    targets = 1:H
    for (h in targets) {
        predobj <- predcov(res, ahead = h)
        pred_covs[, , h] <- apply(predobj, c(1, 2), mean)
    }
    ''')
    
    pred_covs = np.array(robjects.r('pred_covs'))  # N, N, H

    for h, date in enumerate(oos_dates):
        Ht = pred_covs[:, :, h]
        mu = mean_matrix.loc[date, stock_list].values
        np.save(f'{save_path}/{date}_mean', mu)
        np.save(f'{save_path}/{date}_cov', Ht)
