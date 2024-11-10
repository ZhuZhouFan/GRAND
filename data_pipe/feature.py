import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# import warnings
import argparse
from joblib import Parallel, delayed

# warnings.filterwarnings('ignore')

def isnumber(x):
    try:
        float(x)
        return False
    except:
        return True
    
def min_max_normal(df, 
                   start_time='2010-01-01',
                   end_time='2018-12-31'):
    max_value = df.loc[start_time:end_time, :].max()
    min_value = df.loc[start_time:end_time, :].min()
    return (df - min_value)/(max_value - min_value)

def obtain_valid_stock_list(kline_day_path, start_time, end_time,
                            null_patience = 0.20):
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
    
def extract_feature(stock_name, date, previous_date, data_path, index_data, overall_description):
    kline_data = pd.read_csv(
        f'{data_path}/kline_week/{stock_name}.csv', index_col='date')
    basic_data = pd.read_csv(
        f'{data_path}/basic_factor/{stock_name}.csv', index_col='date')
    macro_data = pd.read_csv(
        f'{data_path}/macro_data/macro_economy_week.csv', index_col='date')

    kline_data['ret'] = kline_data['close']/kline_data['close'].shift(1) - 1
    kline_data['log_ret'] = np.log(kline_data['close']) - np.log(kline_data['close'].shift(1))
    kline_data['log_volume'] = np.log(kline_data['volume'] + 1e-6)
    kline_data['log_total_turnover'] = np.log(kline_data['total_turnover'] + 1e-6)
    kline_data['log_num_trades'] = np.log(kline_data['num_trades'] + 1e-6)
    
    kline_data['index_ret'] = index_data['close']/index_data['close'].shift(1) - 1
    kline_data['index_log_volume'] = np.log(index_data['volume'] + 1e-6)
    kline_data['index_log_total_turnover'] = np.log(index_data['total_turnover'] + 1e-6)
    kline_data['excess_ret'] = kline_data['ret'] - kline_data['index_ret']
    
    # apply logorithm operation to some big values
    basic_data['a_share_market_val_in_circulation'] = np.log(basic_data['a_share_market_val_in_circulation'] + 1e-6)
    
    # volume, total_turnover, num_trades
    kline_factor_list = ['open', 'high', 'close', 'low', 'log_volume',
                         'log_total_turnover', 'log_num_trades', 'log_ret', 'excess_ret',
                         'index_ret', 'index_log_volume', 'index_log_total_turnover']
    # a_share_market_val_in_circulation
    basic_factor_list = ['a_share_market_val_in_circulation', 'du_return_on_equity_ttm',
                         'inc_revenue_ttm', 'total_asset_turnover_ttm', 'debt_to_asset_ratio_ttm']
    # macro economy
    macro_factor_list = ['treasury_bond', 'industrial', 'social_finance']
    # sector dummy
    sector_dummy = ['Financials', 'RealEstate', 'HealthCare', 'Industrials', 'Materials', 'ConsumerDiscretionary',
                    'ConsumerStaples', 'InformationTechnology', 'Utilities', 'TelecommunicationServices', 'Energy']
    # insert value
    selected_data = pd.DataFrame(columns=kline_factor_list + basic_factor_list + macro_factor_list + sector_dummy,
                                 index = index_data.loc[previous_date:date, :].index.values)
    
    kline_data[kline_factor_list] = min_max_normal(kline_data[kline_factor_list])
    basic_data[basic_factor_list] = min_max_normal(basic_data[basic_factor_list])
    macro_data[macro_factor_list] = min_max_normal(macro_data[macro_factor_list])
    
    selected_data[kline_factor_list] = kline_data.loc[previous_date:date, kline_factor_list]
    selected_data[basic_factor_list] = basic_data.loc[previous_date:date, basic_factor_list]
    selected_data[macro_factor_list] = macro_data.loc[previous_date:date, macro_factor_list]

    sector_code = overall_description.at[stock_name, 'sector_code']
    selected_data[sector_dummy] = 0
    if sector_code == 'Unknown':
        pass
    else:
        selected_data[sector_code] = 1
    
    # remove abnormal value
    selected_data[selected_data.map(isnumber)] = np.nan
    selected_data[np.isinf(selected_data)] = np.nan
    # deal with nan
    selected_data.ffill(inplace=True)
    selected_data.fillna(0.5, inplace=True)
    return selected_data


def one_day(date, data_path, save_path, index_data, 
            stock_list, overall_description, date_array,
            num_worker=20):
    previous_date = date_array[np.where(date_array == date)[0].item() - lag_order + 1]

    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_feature)
                                               (stock_name, date, previous_date, data_path, index_data, overall_description)
                                               for stock_name in stock_list)

    feature_tensor = np.zeros([stock_list.shape[0], lag_order, P])
    
    for i in range(len(one_day_list)):
        if one_day_list[i].values.shape[0] < lag_order:
            tem = one_day_list[i].values.shape[0]
            feature_tensor[i, -tem:, :] = one_day_list[i].values
        else:
            feature_tensor[i, :, :] = one_day_list[i].values 

    np.save(f'{save_path}/{date}/feature.npy', feature_tensor)


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
    valid_time = '2018-12-31'
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
    stock_list = obtain_valid_stock_list(f'{data_path}/kline_day', start_time, valid_time, 0.20)

    for date in tqdm(normal_week_array[lag_order:], desc='construct feature'):
        date_save_path = f'{save_path}/{date}'
        if (os.path.exists(f'{date_save_path}/feature.npy') & skip_exsting):
            continue
        elif not os.path.exists(date_save_path):
            os.makedirs(date_save_path)

        one_day(date, data_path, save_path, index_week_data,
                stock_list, overall_description, normal_week_array,
                num_worker=50)