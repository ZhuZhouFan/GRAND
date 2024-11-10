import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import argparse

warnings.filterwarnings('ignore')

def isnumber(x):
    try:
        float(x)
        return False
    except:
        return True
    
def obtain_valid_stock_list(kline_day_path, start_time, end_time,
                            null_patience = 0.15):
    existing_stock_list = os.listdir(kline_day_path)
    existing_stock_list = [x[:-4] for x in existing_stock_list]
    existing_stock_list.sort()
    
    valid_stock_list = []
    for stock_name in existing_stock_list:
        df = pd.read_csv(f'{kline_day_path}/{stock_name}.csv', index_col = 'date')
        df.loc[(df['volume'] < 1)|(df['total_turnover'] < 1), :] = np.nan
        df = df.loc[start_time:end_time, :]
        if (df.isna().sum().max()) < (null_patience * df.shape[0]):
            valid_stock_list.append(stock_name)
    return np.array(valid_stock_list)


def extract_label(stock_name, buy_date, sell_date, data_path, index_data):
    kline_data = pd.read_csv(
        f'{data_path}/kline_week/{stock_name}.csv', index_col='date')
    try:
        label = np.log(kline_data.at[sell_date, 'close']) - np.log(kline_data.at[buy_date, 'close']) 
        if np.isnan(label):
            label = 0.0
    except KeyError as e:
        label = 0.0
    return label

def one_day(date, data_path, save_path, index_data, stock_list,
            weekly_date_array, horizon = 1, num_worker=20):
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
    parser.add_argument('--lag', type=int, 
                         help='Number of lagged value of each feature (S in the paper).')
    parser.add_argument('--data-folder', type=str, 
                         help='Path to your data folder')
    args = parser.parse_args()

    data_path = args.data_folder
    lag_order = args.lag
    
    start_time = '2010-01-01'
    end_time = '2022-07-31'
    horizon = 1
    P = 31
    skip_exsting = True
    save_path = f'{data_path}/tensor/lag_{lag_order}_horizon_{horizon}'
    overall_description = pd.read_csv(f'{data_path}/overall_description.csv', index_col='order_book_id')
    sector_code_list = overall_description['sector_code'].unique().tolist()
    industry_code_list = overall_description['industry_code'].unique().tolist()

    index_week_data = pd.read_csv(f'{data_path}/kline_week_index/000001.XSHG.csv', index_col='date')
    normal_week_array = (index_week_data.loc[start_time: end_time, :].index.values)
    stock_list = obtain_valid_stock_list(f'{data_path}/kline_day', start_time, end_time)

    for date in tqdm(normal_week_array[lag_order:-1], desc='construct label'):
        date_save_path = f'{save_path}/{date}'
        if (os.path.exists(f'{date_save_path}/label.npy') & skip_exsting):
            continue
        elif not os.path.exists(date_save_path):
            os.makedirs(date_save_path)

        one_day(date, data_path, save_path, index_week_data, 
                stock_list, normal_week_array, horizon, 30)