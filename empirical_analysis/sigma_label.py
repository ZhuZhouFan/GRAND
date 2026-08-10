"""
Construct variance labels for the SGA-based sigma model.

Inputs: QCM moment CSVs under ``{project_path}/moment/.../``; requires
``--lr``, ``--hidden``, and ``--lag`` to match the upstream quantile run.
Operations: for each date, read the horizon-aligned QCM variance target for
every valid stock (missing values filled with 0).
Outputs: ``{project_path}/sigma_tensor/hidden_*_lr_*_lag_*_horizon_*/{date}/label.npy``
with shape ``[N, 1]``.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
from joblib import Parallel, delayed

import sys
sys.path.append('.')
from config import project_path, start_time, end_time, horizon, lag

warnings.filterwarnings('ignore')

def extract_label(stock_name, sell_date, data_path, moment_path):
    basic_data = pd.read_csv(f'{data_path}/basic_factor/{stock_name}.csv', index_col='date')

    try:
        moment_data = pd.read_csv(f'{moment_path}/{stock_name}.csv', index_col='date')
        moment_data = moment_data.shift(1)
    except FileNotFoundError:
        moment_data = basic_data.copy()
        moment_data['variance'] = 0.0

    # label
    try:
        label = moment_data.at[sell_date, 'variance']
        if np.isnan(label):
            label = 0.0
    except KeyError as e:
        label = 0.0
    return label

def one_day(date, data_path, moment_path, save_path, stock_list,
            weekly_date_array, horizon = 1, num_worker=20):
    buy_date = date
    sell_date = weekly_date_array[np.where(weekly_date_array == date)[0].item() + horizon]

    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_label)
                                               (stock_name, sell_date, data_path, moment_path)
                                               for stock_name in stock_list)

    label_tensor = np.zeros([stock_list.shape[0], 1])

    for i in range(len(one_day_list)):  # type: ignore
        label_tensor[i, 0] = one_day_list[i]

    np.save(f'{save_path}/{date}/label.npy', label_tensor)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate.')
    parser.add_argument('--hidden', type=int, default=128, 
                        help='Number of hidden units in encoder.')
    parser.add_argument('--lag', type=int, default=lag,
                         help='Number of lagged value of each feature (S in the paper).')
    args = parser.parse_args()

    start_time = '2010-01-01'
    end_time = '2022-07-31'
    
    lag_order = args.lag
    hidden_dim = args.hidden
    moment_path = f'{project_path}/moment/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_{horizon}'
    
    skip_exsting = True
    save_path = f'{project_path}/sigma_tensor/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_{horizon}'
    overall_description = pd.read_csv(f'{project_path}/overall_description.csv', index_col='order_book_id')
    sector_code_list = overall_description['sector_code'].unique().tolist()

    index_week_data = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    normal_week_array = (index_week_data.loc[start_time: end_time, :].index.values)
    
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)
    
    for date in tqdm(normal_week_array[lag_order:-1], desc='construct label'):
        date_save_path = f'{save_path}/{date}'
        if (os.path.exists(f'{date_save_path}/label.npy') & skip_exsting):
            continue
        elif not os.path.exists(date_save_path):
                os.makedirs(date_save_path)

        one_day(date, project_path, moment_path, save_path, stock_list, normal_week_array, horizon, 30)