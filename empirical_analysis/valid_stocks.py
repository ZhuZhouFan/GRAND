"""
Build the universe of stocks used throughout the GRAND empirical pipeline.

Inputs: daily kline CSVs under ``{project_path}/kline_day/``.
Operations: drop stocks with excessive missing or zero-volume observations over
the in-sample window.
Outputs: ``{project_path}/valid_stocks.npy`` (1-D array of stock IDs).
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('.')
from config import project_path

def obtain_valid_stock_list(kline_day_path, start_time, end_time,
                            null_patience = 0.20):
    """Obtain the valid stocks from the kline_day data."""
    existing_stock_list = os.listdir(kline_day_path)
    existing_stock_list = [x[:-4] for x in existing_stock_list]
    existing_stock_list.sort()
    
    print(f'Existing stocks: {len(existing_stock_list)}')
    
    valid_stock_list = []
    for stock_name in existing_stock_list:
        df = pd.read_csv(f'{kline_day_path}/{stock_name}.csv', index_col = 'date')
        df.loc[(df['volume'] < 1)|(df['total_turnover'] < 1), :] = np.nan
        df = df.loc[start_time:end_time, :]
        if (df.isna().sum().max()) < (null_patience * df.shape[0]):
            valid_stock_list.append(stock_name)
    return np.array(valid_stock_list)

if __name__ == '__main__':
    
    kline_day_path = f'{project_path}/kline_day'
    start_time = '2010-01-01'
    end_time = '2018-12-31'
    valid_stocks = obtain_valid_stock_list(kline_day_path, start_time, end_time)
    
    valid_stocks.sort()
    print(f'Valid stocks from {start_time} to {end_time}: {valid_stocks.shape[0]}')
    np.save(f'{project_path}/valid_stocks.npy', valid_stocks)