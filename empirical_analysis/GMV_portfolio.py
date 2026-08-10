import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import sys
sys.path.append('.')
from config import start_time, valid_time, end_time, project_path


def gmv_weight(cov_mat: np.ndarray):
    N = cov_mat.shape[0]
    ones = np.ones([N, 1])
    optimal_weights = (np.linalg.inv(cov_mat) @ ones) / (ones.transpose() @ np.linalg.inv(cov_mat) @ ones)
    return optimal_weights.squeeze()


def backtest_with_gmv_weights(covariance_dict: dict,
                              label_matrix: pd.DataFrame,
                              stock_list: list,
                              backtest_array: np.array):
    previous_weights = pd.DataFrame(columns=stock_list, index=backtest_array)
    previous_weights.index.name = 'date'
    previous_weights.iloc[0, :] = 0
    current_weights = pd.DataFrame(columns=stock_list, index=backtest_array)
    current_weights.index.name = 'date'

    result_table = pd.DataFrame(columns=['date', 'pnl', 'nav'])
    result_table['date'] = backtest_array
    result_table.set_index('date', inplace=True)

    for i, date in tqdm(enumerate(backtest_array[:-1]), total=len(backtest_array[:-1])):
        next_date = backtest_array[i+1]
        ground_truth = label_matrix.loc[date, stock_list].values

        try:
            cov_mat = covariance_dict[date]
            optimal_weights = gmv_weight(cov_mat)
        except Exception as e:
            print(e)
            optimal_weights = previous_weights.loc[date, :]

        expected_ret = (ground_truth @ optimal_weights)
        result_table.loc[next_date, 'pnl'] = expected_ret

        current_weights.loc[date, stock_list] = optimal_weights
        previous_weights.loc[next_date, stock_list] = optimal_weights

    result_table['nav'] = (1 + result_table['pnl']).cumprod()
    return result_table, current_weights


def compute_pm(table):

    annual_ret = table[f'pnl'].mean() * 52
    annual_risk = (np.sqrt(52) * table[f'pnl'].std())
    ir = annual_ret/annual_risk

    print(f"annual return: {100 * annual_ret:.2f}%")
    print(f"annual risk: {100 * annual_risk:.2f}%")
    print(f"IR: {ir:.2f}")

    return annual_ret, annual_risk, ir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    index_week = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    selected_dates = index_week.loc[start_time:end_time, :].index.values
    oos_dates = index_week.loc[valid_time:end_time, :].index.values
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)

    label_matrix = pd.DataFrame(index=oos_dates, columns=stock_list)
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

    cov_dict = dict.fromkeys(oos_dates, 0)
    for date in oos_dates:
        cov_dict[date] = np.load(f'{load_path}/{date}_cov.npy')

    result_table, current_weights = backtest_with_gmv_weights(cov_dict,
                                                              label_matrix,
                                                              stock_list,
                                                              oos_dates)

    print(f'Backtesting {method_type} method')
    compute_pm(result_table)
    print('*'*20)
    print('\n'*3)

    result_table.to_csv(f'{project_path}/gmv_results/{method_type}.csv')