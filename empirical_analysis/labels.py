"""
Construct forward-return labels for SGA quantile / mean models.

Inputs: ``valid_stocks.npy``, weekly kline CSVs, and the weekly index calendar.
Operations: for each date, compute horizon-ahead close-to-close returns for all
valid stocks (missing values filled with 0).
Outputs: ``{project_path}/tensor/lag_{S}_horizon_{h}/{date}/label.npy`` with
shape ``[N, 1]``.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import sys
sys.path.append('.')
from config import project_path, start_time, end_time, horizon, lag, P

def isnumber(x):
    try:
        float(x)
        return False
    except:
        return True
    
def extract_label(stock_name, buy_date, sell_date, data_path, index_data):
    """Extract the label r_i,t for a given stock at a given date."""
    kline_data = pd.read_csv(
        f'{data_path}/kline_week/{stock_name}.csv', index_col='date')
    try:
        label = kline_data.at[sell_date, 'close'] / kline_data.at[buy_date, 'close'] - 1
        if np.isnan(label):
            label = 0.0
    except KeyError as e:
        label = 0.0
    return label

def one_day(date, data_path, save_path, index_data, stock_list,
            weekly_date_array, horizon = 1, num_worker=20):
    """Extract the cross-sectional labels r_t for a given date."""
    buy_date = date
    sell_date = weekly_date_array[np.where(weekly_date_array == date)[0].item() + horizon]

    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_label)
                                               (stock_name, buy_date, sell_date, data_path, index_data)
                                               for stock_name in stock_list)

    label_tensor = np.zeros([stock_list.shape[0], 1])

    for i in range(len(one_day_list)):  # type: ignore
        label_tensor[i, 0] = one_day_list[i]

    np.save(f'{save_path}/{date}/label.npy', label_tensor)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lag', type=int, default=lag,
                         help='Number of lagged value of each feature (S in the paper).')
    args = parser.parse_args()
    lag_order = args.lag
    
    skip_exsting = True
    save_path = f'{project_path}/tensor/lag_{lag_order}_horizon_{horizon}'

    index_week_data = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    normal_week_array = (index_week_data.loc[start_time: end_time, :].index.values)
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)

    for date in tqdm(normal_week_array[lag_order:-1], desc='construct label'):
        date_save_path = f'{save_path}/{date}'
        if (os.path.exists(f'{date_save_path}/label.npy') & skip_exsting):
            continue
        elif not os.path.exists(date_save_path):
            os.makedirs(date_save_path)

        one_day(date, project_path, save_path, index_week_data, 
                stock_list, normal_week_array, horizon, 50)