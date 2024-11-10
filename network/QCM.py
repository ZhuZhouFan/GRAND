import pandas as pd
from scipy import stats
import numpy as np
import statsmodels.api as sm
from hypothesis_test import Kupic_test, Christofer_test

def re_arrange(stock_name, tau_list, inference_dict):
    tau_dict = dict.fromkeys(tau_list, 0)
    for tau in tau_list:
        inference_df = inference_dict[tau]
        if tau == tau_list[-1]:
            stock_df = inference_df.loc[inference_df['c_code'] == stock_name, ['date', 'c_code', tau, 'ground_truth']]
            tau_dict[tau] = stock_df.copy()
        else:
            stock_df = inference_df.loc[inference_df['c_code'] == stock_name, ['date', 'c_code', tau]]
            tau_dict[tau] = stock_df.copy()
    
    for i, tau in enumerate(tau_list):
        if i == 0:
            stock_quantiles = tau_dict[tau]
        else:
            stock_quantiles = pd.merge(stock_quantiles, tau_dict[tau], on=['date', 'c_code'])
    
    return stock_quantiles

def build_design_matrix(tau_list:list):
    n = len(tau_list)
    matrix = np.ones((n, 4))
    matrix[:, 1] = stats.norm.ppf(tau_list, 0, 1)
    matrix[:, 2] = matrix[:, 1]**2 -1
    matrix[:, 3] = matrix[:, 1]**3 - 3 * matrix[:, 1]
    return matrix

def test(quantile_df, tau_list, size, start_time, valid_time):
    # Kupic and Christerfo test
    selected_tau_list = []
    df_train = quantile_df.loc[start_time:valid_time, :]
    # 这个版本里，确实会有问题。
    # 因为feature中可能出现一个问题，有些股票是在valid_time之后才被选入zz1000的成分股
    # 暂时设了个阈值，起码有30个数据吧，不然检验会失效的很厉害
    # 小于30个数据的直接不做检验了
    if df_train.shape[0] < 30:
        return quantile_df[tau_list + ['ground_truth']].copy()
    for tau in tau_list:
        if tau == 0.0:
            continue
        Kupic_stat = Kupic_test(df_train[tau], df_train['ground_truth'], tau)
        Christofer_stat = Christofer_test(df_train[tau], df_train['ground_truth'], tau)
        if ((Kupic_stat < stats.chi2.isf(size, 1)) | (Christofer_stat < stats.chi2.isf(size, 2))):
            selected_tau_list.append(tau)
    selected_df = quantile_df[selected_tau_list + [0.0, 'ground_truth']].copy()
    return selected_df

def QCM_regression(quantile_df:pd.DataFrame):
    tau_list = [x for x in list(quantile_df.columns) if (x != 'ground_truth') & (x != 0.0)]
    result_table = pd.DataFrame(columns = ['y', 'mean', 'variance', 'skewness', 'kurtosis'])
    X = build_design_matrix(tau_list)
    # X = X[:, [1, 2, 3]] # remove the intercept term
    for index in quantile_df.index:
        y = quantile_df.loc[index, tau_list]
        result_table['mean'] = quantile_df[0.0]
        y = y[tau_list].values.transpose()
        
        model_OLS = sm.OLS(y, X)
        result_OLS = model_OLS.fit()
        
        # includes intercept
        variance = result_OLS.params[1]**2
        skewness = 6 * result_OLS.params[2]/result_OLS.params[1]
        kurtosis = 24 * result_OLS.params[3]/result_OLS.params[1] + 3
        result_table.loc[index, ['y', 'variance', 'skewness', 'kurtosis']] = (quantile_df.at[index, 'ground_truth'], variance, skewness, kurtosis)
    return result_table

def compute_QCM_table(stock_name, inference_dict, 
                      tau_list, size, 
                      save_path = None,
                      tolerance = 20, 
                      start_time = '2017-01-01',
                      valid_time = '2019-01-01'):
    stock_df = re_arrange(stock_name, tau_list, inference_dict)
    stock_df.set_index('date', inplace = True)
    quantile_df = test(stock_df, tau_list, size, start_time, valid_time)
    
    if quantile_df.shape[1] <= tolerance:
        print(f'There are too little taus in {stock_name}')
        return None
    
    # log_ret to ret
    # quantile_df = np.exp(quantile_df) - 1
    
    result_table = QCM_regression(quantile_df)
    if save_path is None:
        return result_table
    else:
        result_table.to_csv(f'{save_path}/{stock_name}.csv', index = True)