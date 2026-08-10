import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import statsmodels.api as sm

import sys
sys.path.append('.')
from network.NCM import cov2cor
from config import project_path, start_time, valid_time, end_time

def get_top_correlated_stocks(stock_id, correlation_matrix, top_n=50):
    correlation_with_stock = correlation_matrix[stock_id].drop(stock_id)
    top_stocks = correlation_with_stock.abs().nlargest(top_n).index
    return top_stocks


def get_connected_stocks(stock_id, adj_mat):
    cond = (adj_mat[stock_id] != 0)
    connected_stocks = adj_mat.index[cond].values
    return connected_stocks


def backtest_equal_weight(signal_matrix: pd.DataFrame,
                          label_matrix: pd.DataFrame,
                          backtest_array: np.array,
                          stock_list: pd.DataFrame,
                          trade_fee: float = 0.002,
                          lower: float = 0.9,
                          upper: float = 1.0):
    previous_weights = pd.DataFrame(
        columns=label_matrix.columns, index=backtest_array)
    current_weights = pd.DataFrame(
        columns=label_matrix.columns, index=backtest_array)
    previous_weights.loc[:, :] = 0.0
    current_weights.loc[:, :] = 0.0
    result_table = pd.DataFrame(
        columns=['date', 'pnl', 'turnover', 'cost', 'nav'])
    result_table['date'] = backtest_array
    result_table.set_index('date', inplace=True)

    for i, date in enumerate(backtest_array[:-1]):
        next_date = backtest_array[i+1]

        if i == 0:
            result_table.loc[date, 'pnl'] = 0
            result_table.loc[date, 'turnover'] = 0
            result_table.loc[date, 'cost'] = 0
            result_table.loc[date, 'nav'] = 1

        stock_inter = np.intersect1d(stock_list, signal_matrix.columns)
        signal_vec = signal_matrix.loc[date, stock_inter]
        ground_truth = label_matrix.loc[date, stock_inter]

        lower_bound = signal_vec.quantile(lower)
        upper_bound = signal_vec.quantile(upper)

        selected_stocks = signal_vec[((signal_vec > lower_bound) & (
            signal_vec < upper_bound)) == True].index
        current_weights.loc[date, selected_stocks] = 1/selected_stocks.shape[0]
        expected_ret = (
            current_weights.loc[date, selected_stocks] * ground_truth[selected_stocks]).sum()

        turnover = np.abs(
            current_weights.loc[date, :] - previous_weights.loc[date, :]).sum()
        cost = turnover * trade_fee
        result_table.loc[next_date, 'pnl'] = expected_ret - cost
        result_table.loc[next_date, 'turnover'] = turnover
        result_table.loc[next_date, 'cost'] = cost

        previous_weights.loc[next_date,
                             :] = current_weights.loc[date, :].values

    result_table['nav'] = (1+result_table['pnl']).cumprod()
    return result_table, current_weights


def backtest_equal_weight_HL_portfolio(signal_matrix: pd.DataFrame,
                                       label_matrix: pd.DataFrame,
                                       backtest_array: np.array,
                                       stock_list: pd.DataFrame,
                                       trade_fee: float = 0.0):

    previous_weights = pd.DataFrame(
        columns=label_matrix.columns, index=backtest_array)
    current_weights = pd.DataFrame(
        columns=label_matrix.columns, index=backtest_array)
    previous_weights.loc[:, :] = 0.0
    current_weights.loc[:, :] = 0.0
    result_table = pd.DataFrame(
        columns=['date', 'pnl', 'turnover', 'cost', 'nav'])
    result_table['date'] = backtest_array
    result_table.set_index('date', inplace=True)

    for i, date in enumerate(backtest_array[:-1]):
        next_date = backtest_array[i+1]

        if i == 0:
            result_table.loc[date, 'pnl'] = 0
            result_table.loc[date, 'turnover'] = 0
            result_table.loc[date, 'cost'] = 0
            result_table.loc[date, 'nav'] = 1

        stock_inter = np.intersect1d(stock_list, signal_matrix.columns)
        signal_vec = signal_matrix.loc[date, stock_inter]
        ground_truth = label_matrix.loc[date, stock_inter]

        lower_point = signal_vec.quantile(0.1)
        upper_point = signal_vec.quantile(0.9)

        short_stocks = signal_vec[signal_vec < lower_point].index
        long_stocks = signal_vec[signal_vec > upper_point].index
        selected_stocks = np.hstack([short_stocks, long_stocks])

        current_weights.loc[date, short_stocks] = -1/short_stocks.shape[0]
        current_weights.loc[date, long_stocks] = 1/long_stocks.shape[0]

        expected_ret = (
            current_weights.loc[date, selected_stocks] * ground_truth[selected_stocks]).sum()

        turnover = np.abs(
            current_weights.loc[date, :] - previous_weights.loc[date, :]).sum()
        cost = turnover * trade_fee
        result_table.loc[next_date, 'pnl'] = expected_ret - cost
        result_table.loc[next_date, 'turnover'] = turnover
        result_table.loc[next_date, 'cost'] = cost

        previous_weights.loc[next_date,
                             :] = current_weights.loc[date, :].values

    result_table['nav'] = (1+result_table['pnl']).cumprod()
    return result_table, current_weights


def deciles_backtest(signal_matrix: pd.DataFrame,
                     label_matrix: pd.DataFrame,
                     backtest_dates: np.ndarray,
                     stock_list: np.ndarray,
                     trade_fee: float):
    decile_table = pd.DataFrame(columns=[f'pnl_{decile}' for decile in range(
        10)] + [f'nav_{decile}' for decile in range(10)] + [f'turnover_{decile}' for decile in range(10)])
    for j, (lower, upper) in enumerate([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3),
                                        (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                                        (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]):
        tem_table, weight_table = backtest_equal_weight(signal_matrix,
                                                        label_matrix,
                                                        backtest_dates,
                                                        stock_list,
                                                        trade_fee,
                                                        lower,
                                                        upper)
        decile_table[f'pnl_{j}'] = tem_table['pnl']
        decile_table[f'nav_{j}'] = tem_table['nav']
        decile_table[f'turnover_{j}'] = tem_table['turnover']

    hl_table, hl_weight_table = backtest_equal_weight_HL_portfolio(signal_matrix,
                                                                   label_matrix,
                                                                   backtest_dates,
                                                                   stock_list,
                                                                   trade_fee)

    decile_table[['pnl_HL', 'nav_HL', 'turnover_HL']
                 ] = hl_table[['pnl', 'nav', 'turnover']]

    return decile_table


def MaxDrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) -
                  return_list)/np.maximum.accumulate(return_list))
    if i == 0:
        return 0, 0, 0
    j = np.argmax(return_list[:i])
    return (return_list[j] - return_list[i]) / return_list[j], j, i


def compute_pm(table, risk_free, decile='HL'):
    win_rate = (table[f'pnl_{decile}'] > 0).sum()/table.shape[0]
    annual_ret = table[f'pnl_{decile}'].mean() * 52
    annual_risk = (np.sqrt(52) * table[f'pnl_{decile}'].std())
    sharpe_ratio = (annual_ret - risk_free)/annual_risk
    max_dd = MaxDrawdown(table[f'nav_{decile}'].fillna(1).values)[0]
    turnover = table[f'turnover_{decile}'].mean()
    calmar_ratio = annual_ret/max_dd

    print(f'interested decile: {decile}')
    print(f"annual return: {100 * annual_ret:.2f}%")
    print(f"annual risk: {100 * annual_risk:.2f}%")
    print(f"win rate: {100 * win_rate:.2f}%")
    print(f"maxDD: {100 * max_dd:.2f}%")
    print(f"turnover: {100 * turnover:.2f}%")
    print(f"calmar ratio: {calmar_ratio:.2f}")
    print(f"SR: {sharpe_ratio:.2f}")

    return 100 * annual_ret, 100 * annual_risk, 100 * win_rate, 100 * max_dd, turnover, calmar_ratio, sharpe_ratio


if __name__ == '__main__':
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)
    index_week = pd.read_csv(
        f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    selected_dates = index_week.loc[start_time:end_time, :].index.values
    oos_dates = selected_dates[(selected_dates >= valid_time) & (
        selected_dates <= end_time)]

    ret_matrix = pd.DataFrame(index=selected_dates, columns=stock_list)
    for stock in stock_list:
        try:
            kline_week = pd.read_csv(
                f'{project_path}/kline_week/{stock}.csv', index_col='date')
        except FileNotFoundError:
            continue
        kline_week['return'] = kline_week['close'] / kline_week['close'].shift(1) - 1
        # kline_week['return'] = np.log(kline_week['close']/kline_week['close'].shift(1))
        ret_matrix[stock] = kline_week['return']

    aver_risk_free = 0.023
    trade_rate = 0.00

    label_matrix = ret_matrix.shift(-1)
    backtest_dates = oos_dates[1:-1]

    os.makedirs(f'{project_path}/pair_trading', exist_ok=True)

    if not os.path.exists(f'{project_path}/pair_trading/chen_retdiff.csv'):
        chen_mat = pd.DataFrame(columns=stock_list, index=oos_dates)
        chen_mat.index.name = 'date'
        for date in tqdm(oos_dates):

            ly_date = f'{pd.to_datetime(date).year}-01-01'
            prev_date = (pd.to_datetime(ly_date) -
                         pd.DateOffset(years=4)).strftime('%Y-%m-%d')
            cor_mat = ret_matrix.loc[prev_date:ly_date, :].corr()

            for stock in stock_list:
                pair_stocks = get_top_correlated_stocks(
                    stock, cor_mat, top_n=50)

                pairs_returns = ret_matrix.loc[prev_date:ly_date, pair_stocks]
                stock_returns = ret_matrix.loc[prev_date:ly_date, stock]
                Cret = pairs_returns.mean(axis=1)
                beta_c = stock_returns.cov(Cret)/Cret.var()

                RetDiff = beta_c * \
                    ret_matrix.loc[date, pair_stocks].mean() - \
                    ret_matrix.loc[date, stock]

                chen_mat.loc[date, stock] = RetDiff
        chen_mat.to_csv(f'{project_path}/pair_trading/chen_retdiff.csv')
    else:
        chen_mat = pd.read_csv(f'{project_path}/pair_trading/chen_retdiff.csv', index_col='date')
            
    chen_table = deciles_backtest(chen_mat, label_matrix, backtest_dates, stock_list, trade_rate)
    print('Backtesting Chen method')
    compute_pm(chen_table, aver_risk_free, decile='HL')
    print('*'*20)
    compute_pm(chen_table, aver_risk_free, decile='9')
    print('\n'*3)

    for method_type in ['GRAND', 'GRAND_DCC_GARCH', 'SFM',
                        'FF3', 'FF5', 'FF6', 'CH4', 'HXZ4', 'PCA',
                        'FF3_DCC_GARCH', 'FF5_DCC_GARCH', 'FF6_DCC_GARCH',
                        'CH4_DCC_GARCH', 'HXZ4_DCC_GARCH', 'PCA_DCC_GARCH']:
        # for method_type in ['GRAND']:
        if not os.path.exists(f'{project_path}/pair_trading/{method_type}_retdiff.csv'):
            fac_mat = pd.DataFrame(columns=stock_list, index=oos_dates)
            fac_mat.index.name = 'date'

            if method_type == 'GRAND':
                cov_path = f'{project_path}/sigma_graph_cov/hidden_128_lr_0.0001_lag_48_horizon_1'
            elif method_type == 'SFM':
                cov_path = f'{project_path}/baselines/{method_type}/4_400'
            else:
                cov_path = f'{project_path}/baselines/{method_type}'

            for date in tqdm(oos_dates):

                ly_date = f'{pd.to_datetime(date).year}-01-01'
                prev_date = (pd.to_datetime(ly_date) -
                             pd.DateOffset(years=4)).strftime('%Y-%m-%d')
                cov_mat = np.load(f'{cov_path}/{date}_cov.npy')
                cor_mat = cov2cor(cov_mat)
                cor_mat = pd.DataFrame(
                    cor_mat, index=stock_list, columns=stock_list)

                for stock in stock_list:
                    pair_stocks = get_top_correlated_stocks(
                        stock, cor_mat, top_n=50)

                    pairs_returns = ret_matrix.loc[prev_date:ly_date, pair_stocks]
                    stock_returns = ret_matrix.loc[prev_date:ly_date, stock]
                    Cret = pairs_returns.mean(axis=1)
                    beta_c = stock_returns.cov(Cret)/Cret.var()

                    RetDiff = beta_c * \
                        ret_matrix.loc[date, pair_stocks].mean(
                        ) - ret_matrix.loc[date, stock]

                    fac_mat.loc[date, stock] = RetDiff
            fac_mat.to_csv(
                f'{project_path}/pair_trading/{method_type}_retdiff.csv')
        else:
            fac_mat = pd.read_csv(
                f'{project_path}/pair_trading/{method_type}_retdiff.csv', index_col='date')

        factor_table = deciles_backtest(
            fac_mat, label_matrix, backtest_dates, stock_list, trade_rate)
        print(f'Backtesting {method_type} method')
        compute_pm(factor_table, aver_risk_free, decile='HL')
        print('*'*20)
        compute_pm(factor_table, aver_risk_free, decile='9')
        print('\n'*3)

    for graph_type in ['V', 'E', 'M', 'Q']:
        if not os.path.exists(f'{project_path}/pair_trading/{graph_type}_group_retdiff.csv'):

            graph = np.load(f'{project_path}/{graph_type}_graph.npy', allow_pickle=True).item()
        
            group_mat = pd.DataFrame(columns=stock_list, index=oos_dates)
            group_mat.index.name = 'date'
            for date in tqdm(oos_dates):

                ly_date = f'{pd.to_datetime(date).year}-01-01'
                prev_date = (pd.to_datetime(ly_date) -
                             pd.DateOffset(years=4)).strftime('%Y-%m-%d')
                E_adj = pd.DataFrame(graph[date], index=stock_list, columns=stock_list)

                for stock in stock_list:
                    pair_stocks = get_connected_stocks(stock, E_adj)

                    pairs_returns = ret_matrix.loc[prev_date:ly_date, pair_stocks]
                    stock_returns = ret_matrix.loc[prev_date:ly_date, stock]
                    Cret = pairs_returns.mean(axis=1)
                    beta_c = stock_returns.cov(Cret)/Cret.var()

                    RetDiff = beta_c * \
                        ret_matrix.loc[date, pair_stocks].mean(
                        ) - ret_matrix.loc[date, stock]

                    group_mat.loc[date, stock] = RetDiff
            group_mat.to_csv(f'{project_path}/pair_trading/{graph_type}_group_retdiff.csv')
        else:
            group_mat = pd.read_csv(f'{project_path}/pair_trading/{graph_type}_group_retdiff.csv', index_col='date')

        group_table = deciles_backtest(group_mat, label_matrix, backtest_dates, stock_list, trade_rate)
        print(f'Backtesting GRAND method with {graph_type}_GRAPH')
        compute_pm(group_table, aver_risk_free, decile='HL')
        print('*'*20)
        compute_pm(group_table, aver_risk_free, decile='9')
        print('\n'*3)
