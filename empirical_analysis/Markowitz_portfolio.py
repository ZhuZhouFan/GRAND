"""
Evaluate out-of-sample Markowitz (mean-variance) portfolio performance.

Specifically, it constructs mean-variance portfolios using the predicted conditional
means and covariance matrices, subject to individual asset-weight constraints. It
then performs weekly portfolio rebalancing with transaction costs and reports key
performance measures, including annualized return and risk, Sharpe ratio, maximum
drawdown, turnover, win rate, and Calmar ratio.
"""

import cvxopt as opt
import numpy as np
import pandas as pd
from cvxopt import solvers
import os
from tqdm import tqdm
import argparse

import sys
sys.path.append('.')
from config import start_time, valid_time, end_time, project_path

solvers.options['show_progress'] = False


def optimal_portfolio(ret_vec: np.ndarray,
                      cov_mat: np.ndarray,
                      max_weight: float = 0.1,
                      min_weight: float = 0.0,
                      iter_num: int = 100,
                      quadratic: bool = True) -> dict:
    solvers.options['show_progress'] = False
    n = ret_vec.shape[0]
    P = opt.matrix(cov_mat)
    q = opt.matrix(np.zeros(n))
    
    G = opt.matrix(np.vstack([-1 * np.eye(n), np.eye(n)]))
    h = opt.matrix(np.vstack([np.ones((n, 1)) * -1 * min_weight, np.ones((n, 1)) * max_weight]))
    A = opt.matrix(np.vstack([ret_vec.reshape(1, -1), np.ones((1, n))]))
    
    target_rets = [np.quantile(ret_vec, x) for x in np.linspace(0, 1, iter_num, endpoint=False)]
    
    feasible_rets = []
    feasible_risks = []
    feasible_weights = []
    for target_return in target_rets:
        try:
            b = opt.matrix(np.array([target_return, 1.0]))
            sol = solvers.qp(P, q, G, h, A, b)
            weight = np.array(sol['x'])
            risk = np.sqrt(weight.T @ cov_mat @ weight).item()

            feasible_rets.append(target_return)
            feasible_risks.append(risk)
            feasible_weights.append(weight.squeeze())
        except Exception as e:
            continue
    
    feasible_rets = np.array(feasible_rets)
    feasible_risks = np.array(feasible_risks)
    
    success_flag = False
    
    if quadratic:
        cond = (feasible_rets >= feasible_rets[feasible_risks.argmin()])
        upper_rets = feasible_rets[cond]
        upper_risks = feasible_risks[cond]
    
        try:
            coefs = np.polyfit(upper_risks, upper_rets, 2)
            opt_risk = np.sqrt(coefs[2] / coefs[0])
            opt_ret = coefs[0] * opt_risk**2 + coefs[1] * opt_risk + coefs[2]

            b_opt = opt.matrix(np.array([opt_ret, 1.0]))
            opt_sol = solvers.qp(P, q, G, h, A, b_opt)
            opt_weight = np.array(opt_sol['x']).squeeze()
            
            success_flag = True
        except Exception as e:
            success_flag = False
    
    if (not success_flag) or (not quadratic):
        sample_opt = (np.array(feasible_rets)/np.array(feasible_risks)).argmax()
        opt_weight = feasible_weights[sample_opt]
        opt_risk = feasible_risks[sample_opt]
        opt_ret = feasible_rets[sample_opt]
        coefs = None
    
    solution = {'opt_weight': opt_weight,
                'opt_ret': opt_ret,
                'opt_risk': opt_risk,
                'feasible_weights': feasible_weights,
                'feasible_rets': feasible_rets,
                'feasible_risks': feasible_risks,
                'efficient_coefs': coefs}
    
    return solution

def backtest_with_optimal_weights(mean_matrix:pd.DataFrame,
                                  covariance_dict:dict,
                                  label_matrix:pd.DataFrame,
                                  stock_list:list, 
                                  backtest_array:np.array,
                                  min_weight:float = 0.0,
                                  max_weight:float = 0.1,
                                  iter_num:int = 10,
                                  trade_fee:float = 0.002):
    previous_weights = pd.DataFrame(columns=mean_matrix.columns, index = backtest_array)
    previous_weights.index.name = 'date'
    previous_weights.iloc[0, :] = 0
    current_weights = pd.DataFrame(columns=mean_matrix.columns, index = backtest_array)
    current_weights.index.name = 'date'
    
    result_table = pd.DataFrame(columns = ['date', 'ret', 'pnl', 'turnover', 'cost', 'nav'])
    result_table['date'] = backtest_array
    result_table.set_index('date', inplace = True)
    
    for i, date in tqdm(enumerate(backtest_array[:-1]), total = len(backtest_array[:-1])):
        next_date = backtest_array[i+1]
        
        mean_vec = mean_matrix.loc[date, stock_list].values.astype(float)
        mean_vec = np.expand_dims(mean_vec, axis = 1)
        ground_truth = label_matrix.loc[date, stock_list].values
        
        try:
            cov_mat = covariance_dict[date]
            
            sol = optimal_portfolio(mean_vec, cov_mat, 
                                    max_weight = max_weight,
                                    min_weight = min_weight,
                                    iter_num=iter_num,
                                    quadratic=True)
            optimal_weights = sol['opt_weight']
            current_weights.loc[date, stock_list] = optimal_weights
            
        except Exception as e:
            print(e)
            optimal_weights = previous_weights.loc[date, :].values
        
        expected_ret = ground_truth @ optimal_weights
        turnover = np.abs(current_weights.loc[date, :] - previous_weights.loc[date, :]).sum()
        cost = turnover * trade_fee
        result_table.loc[next_date, 'ret'] = expected_ret
        result_table.loc[next_date, 'pnl'] = expected_ret - cost
        result_table.loc[next_date, 'turnover'] = turnover
        result_table.loc[next_date, 'cost'] = cost
            
        previous_weights.loc[next_date, stock_list] = optimal_weights
    result_table['nav'] = (1 + result_table['pnl']).cumprod()
    return result_table, current_weights

def MaxDrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list)- return_list)/np.maximum.accumulate(return_list))
    if i == 0:
        return 0, 0, 0
    j = np.argmax(return_list[:i])
    return(return_list[j] - return_list[i]) / return_list[j],j,i

def compute_pm(table, risk_free = 0.023):
    win_rate = (table[f'pnl'] > 0).sum()/table.shape[0]
    annual_ret = table[f'pnl'].mean() * 52
    annual_risk = (np.sqrt(52) * table[f'pnl'].std())
    sharpe_ratio = (annual_ret - risk_free)/annual_risk
    max_dd = MaxDrawdown(table[f'nav'].fillna(1).values)[0]
    turnover = table[f'turnover'].mean()
    calmar_ratio = annual_ret/max_dd
    
    # 打印各项指标
    print(f"annual return: {100 * annual_ret:.2f}%")
    print(f"annual risk: {100 * annual_risk:.2f}%")
    print(f"win rate: {100 * win_rate:.2f}%")
    print(f"maxDD: {100 * max_dd:.2f}%")
    print(f"turnover: {100 * turnover:.2f}%")
    print(f"calmar ratio: {calmar_ratio:.2f}")
    print(f"SR: {sharpe_ratio:.2f}")
    
    return 100 * annual_ret, 100 * annual_risk, 100 * win_rate, 100 * max_dd, 100 * turnover, calmar_ratio, sharpe_ratio

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type = float, default=0.05)
    parser.add_argument('--min', type = float, default=0.0)
    
    parser.add_argument('--method', type = str)
    args = parser.parse_args()
    
    index_week = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col = 'date')
    selected_dates = index_week.loc[start_time:end_time, :].index.values
    oos_dates = index_week.loc[valid_time:end_time, :].index.values
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)
    aver_risk_free = 0.023
    
    label_matrix = pd.DataFrame(index = oos_dates, columns=stock_list)
    for stock in stock_list:
        try:
            kline_week = pd.read_csv(f'{project_path}/kline_week/{stock}.csv', index_col='date')
        except FileNotFoundError:
            continue
        kline_week['return'] = kline_week['close'].shift(-1)/kline_week['close'] - 1
        label_matrix[stock] = kline_week['return']
        
    method_type = args.method
    
    if method_type == 'GRAND':
        load_path = f'{project_path}/sigma_graph_cov/hidden_128_lr_0.0001_lag_48_horizon_1'
    else:
        raise ValueError(f'Specify the loading path of the method: {method_type}')

    mean_matrix = pd.DataFrame(index = oos_dates, columns=stock_list)
    for date in oos_dates:
        mean_vec = np.load(f'{load_path}/{date}_mean.npy')
        if np.isnan(mean_vec).sum() > 0:
            mean_vec[np.isnan(mean_vec)] = 0.0
        mean_matrix.loc[date, :] = mean_vec
        
    cov_dict = dict.fromkeys(oos_dates, 0)
    for date in oos_dates:
        cov_dict[date] = np.load(f'{load_path}/{date}_cov.npy')
                
    result_table, current_weights = backtest_with_optimal_weights(mean_matrix,
                                                                  cov_dict,
                                                                  label_matrix,
                                                                  stock_list,
                                                                  oos_dates,
                                                                  min_weight=args.min,
                                                                  max_weight=args.max,
                                                                  iter_num=10)
     
     
    print(f'Backtesting {method_type} method')
    compute_pm(result_table, aver_risk_free)
    print('*'*20)
    print('\n'*3)
    
    os.makedirs(f'{project_path}/MM_backtest_results/{args.min}_{args.max}', exist_ok=True)
     
    result_table.to_csv(f'{project_path}/MM_backtest_results/{args.min}_{args.max}/{method_type}.csv')