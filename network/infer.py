import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import sys
import argparse
from model import SGA
from train import fetch_basic_attributes
from QCM import compute_QCM_table

sys.path.append('.')
from data_pipe.feature import obtain_valid_stock_list

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def infer(N, P, hidden_dim, tau, 
          date_array, stock_list,
          model_path, tensor_path,
          graph=False):
    T = date_array.shape[0]
    K_df = pd.read_csv(f'{model_path}/{tau}/K_result.csv')
    K = K_df.loc[K_df['best_score'].argmin(), 'K']
    network = SGA(individual_num=N,
                  feature_dim=P,
                  K=K,
                  hidden_dim=hidden_dim,
                  num_layers=2)
    network.to(network.device)
    
    network.load_state_dict(torch.load(f'{model_path}/{tau}/{K}/network_best.pth',
                                       map_location=network.device,
                                       weights_only=True))
    network.eval()

    result_dict = dict.fromkeys(date_array, 0)
    mat_dict = dict.fromkeys(date_array, 0)

    for date in date_array:
        result_dict[date] = pd.DataFrame(
            columns=['date', 'c_code', tau, 'ground_truth'])
        result_dict[date]['c_code'] = stock_list
        result_dict[date]['date'] = date

    with torch.no_grad():
        for i in range(T):
            date = date_array[i]
            feature_tensor = np.load(f'{tensor_path}/{date}/feature.npy')
            X = torch.Tensor(feature_tensor).to(network.device)
            network_output = network.forward(X)
            result_dict[date][tau] = network_output.cpu().numpy()
            try:
                label_tensor = np.load(f'{tensor_path}/{date}/label.npy')
                result_dict[date]['ground_truth'] = label_tensor[:, 0]
            except FileNotFoundError:
                result_dict[date]['ground_truth'] = np.nan
            if graph:
                mat_dict[date] = network.encoder_graph(X).cpu().numpy()

    result_df = pd.concat(result_dict.values(), axis=0)
    return result_df, mat_dict, K


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, 
                        help='Learning rate.')
    parser.add_argument('--lag', type=int, 
                        help='Number of lagged value of each feature')
    parser.add_argument('--hidden', type=int, 
                        help='Number of hidden units in encoder.')

    parser.add_argument('--data-folder', type=str,
                        help='Path to your data folder')
    parser.add_argument('--start-time', type=str, default='2010-01-01',
                        help='Beginning of full dataset')
    parser.add_argument('--valid-time', type=str, default='2018-12-31',
                        help='Ending of in-sample dataset')
    parser.add_argument('--end-time', type=str, default='2022-07-31',
                        help='Ending of full dataset')

    parser.add_argument('--cuda', type=int, 
                        help='Number of GPU device training on.')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    start_time = args.start_time
    valid_time = args.valid_time
    end_time = args.end_time
    hidden_dim = args.hidden
    data_path = args.data_folder
    model_path = f'{data_path}/quantile_model/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_1'
    tensor_path = f'{data_path}/tensor/lag_{args.lag}_horizon_1'
    moment_path = f'{data_path}/moment/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_1'
    N, P, _ = fetch_basic_attributes(tensor_path, start_time, end_time)
    tolerance = 20
    size = 0.01

    if not os.path.exists(moment_path):
        os.makedirs(moment_path)

    tau_list = os.listdir(model_path)
    tau_list.sort()
    tau_list = [float(x) for x in tau_list if isnumber(x)]

    date_list = os.listdir(tensor_path)
    date_list.sort()
    date_array = np.array(date_list)
    date_array = date_array[(date_array >= start_time) & (date_array <= end_time)]

    stock_list = obtain_valid_stock_list(f'{data_path}/kline_day', start_time, valid_time, 0.20)
    
    inference_dict = dict.fromkeys(tau_list, 0)
    for tau in tqdm(tau_list, desc='inference'):
        result_df, mat_dict, K = infer(N=N,
                                       P=P,
                                       hidden_dim=hidden_dim,
                                       tau=tau,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=model_path,
                                       tensor_path=tensor_path,
                                       graph=False)
        inference_dict[tau] = result_df

    for stock_name in tqdm(stock_list, desc='QCM regression'):
        try:
            df = compute_QCM_table(stock_name,
                                   inference_dict,
                                   tau_list,
                                   tolerance=tolerance,
                                   size=size,
                                   start_time=start_time,
                                   valid_time=valid_time)
            df.to_csv(f'{moment_path}/{stock_name}.csv')
        except Exception as e:
            print(f'{stock_name} {e}')
