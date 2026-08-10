
import numpy as np
import pandas as pd
import os
import argparse
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, end_time

pandas2ri.activate()
numpy2ri.activate()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=str, default='FF3', help='How to demean')
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
    
    save_path = f'{project_path}/{args.mean}_DCC_GARCH'
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
    
    if args.mean == 'PCA':
        alpha_init = 0.00
        beta_init = 0.95
    elif args.mean == 'FF3':
        alpha_init = 0.04
        beta_init = 0.9
    elif args.mean == 'FF5':
        alpha_init = 0.04
        beta_init = 0.9
    elif args.mean == 'GRAND':
        alpha_init = 0.1
        beta_init = 0.85
    
    robjects.globalenv['returns'] = in_sample_data
    robjects.globalenv['alpha_init'] = alpha_init
    robjects.globalenv['beta_init'] = beta_init
    
    robjects.r('''
    library(rugarch)
    library(xdcclarge)

    Rtn <- returns
    n <- dim(Rtn)[2]

    cdcc_estimate_ast <- function(ini.para = c(0.05, 0.93), ht, residuals, method = c("COV", "LS", "NLS"), ts = 1) {
        stdresids <- residuals / sqrt(ht)
        flag <- match.arg(method)

        if (flag == "NLS") {
            print("Non-linear shrinkage Step Now...")
            uncR <- nlshrink::nlshrink_cov(stdresids)
        } else if (flag == "LS") {
            print("Linear shrinkage Step Now...")
            uncR <- nlshrink::linshrink_cov(stdresids)
        } else {
            uncR <- stats::cov(stdresids)
        }

        print("Optimization Step Now...")
        result <- cdcc_optim(param = ini.para, ht = ht, residuals = residuals, stdresids = stdresids, uncR = uncR)
        print("Construction Rt Step Now...")
        cdcc_Rt <- cdcc_correlations(result$par, stdresids, uncR, ts)

        list(result = result, cdcc_Rt = cdcc_Rt, uncR = uncR)
    }

    spec <- ugarchspec(
        variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(0, 0), include.mean=FALSE), 
        distribution.model = "norm",
        start.pars = list(alpha1 = alpha_init, beta1 = beta_init) 
    )
    mspec = multispec(replicate(spec, n = n))
    fitlist = multifit(multispec = mspec, data = Rtn)
    ht <- sigma(fitlist)^2
    residuals <- residuals(fitlist)

    GARCH_params <- sapply(fitlist@fit, function(fit) fit@fit$coef)

    cDCC <- cdcc_estimate_ast(ini.para = c(0.05, 0.93), ht, residuals, "NLS")

    uncR <- cDCC$uncR
    cdcc_params <- cDCC$result$par
    ''')

    insample_ht = pd.DataFrame(robjects.r('ht'))  # T, N
    insample_residual = pd.DataFrame(robjects.r('residuals'))  # N, N
    cdcc_alpha, cdcc_beta = robjects.r('cdcc_params')
    GARCH_params = robjects.r('GARCH_params')  # (mean, intercept, alpha, beta), N
    uncR = robjects.r('uncR')
    
    print(uncR)
    print(cdcc_alpha, cdcc_beta)

    ht = np.zeros([N, T])
    residual = np.zeros([N, T])
    Qt = np.zeros([T, N, N])

    ht[:, :train_size] = insample_ht.values.T
    residual[:, :train_size] = insample_residual.values.T

    demean_matrix_ = demean_ret_matrix.values.T
    for t in range(train_size, T):
        var_prev = ht[:, t - 1]
        res_prev = residual[:, t - 1]

        for i in range(N):
            ht[i, t] = GARCH_params[0, i] + GARCH_params[1, i] * res_prev[i]**2 \
                       + GARCH_params[2, i] * var_prev[i]
            residual[i, t] = demean_matrix_[i, t]

    for t, date in enumerate(demean_ret_matrix.index.values):
        if t == 0:
            Qt[t, :, :] = np.eye(N)
        else:
            s_prev = residual[:, t - 1]/np.sqrt(ht[:, t - 1])
            Qt[t, :, :] = (1 - cdcc_alpha - cdcc_beta) * uncR \
                + cdcc_alpha * np.outer(s_prev, s_prev) \
                + cdcc_beta * Qt[t - 1, :, :]

        D = np.diag(np.diagonal(Qt[t, :, :]))
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diagonal(D)))
        Rt = np.dot(np.dot(D_inv_sqrt, Qt[t, :, :]), D_inv_sqrt)
        
        Ht = np.sqrt(np.diag(ht[:, t])) @ Rt @ np.sqrt(np.diag(ht[:, t]))
        mu = mean_matrix.loc[date, stock_list].values
        
        np.save(f'{save_path}/{date}_mean', mu)
        np.save(f'{save_path}/{date}_cov', Ht)