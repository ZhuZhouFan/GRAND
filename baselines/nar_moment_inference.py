"""
Run NAR quantile inference and construct per-stock QCM moment tables.

Inputs: trained SGA graph checkpoints under ``{project_path}/quantile_model/...``,
NAR checkpoints under ``{project_path}/nar_quantile_model/...``, and feature/label
tensors; requires ``--lr``, ``--hidden``, ``--lag``, and ``--cuda`` matching the
trained model directories.
Operations: load the best SGA ``K`` and NAR weights per ``tau``, predict
quantiles over the full sample window (SGA graph + NAR decoder), screen with
Kupiec/Christoffersen tests, and fit QCM regressions.
Outputs: ``{project_path}/nar_moment/hidden_*_lr_*_lag_*_horizon_{h}/{stock}.csv``
and ``{project_path}/nar_feasible_stock_list.npy``.
"""

import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import argparse

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, end_time, horizon, lag
from network.model import SGA
from network.NAR import NAR
from network.QCM import compute_QCM_table
from empirical_analysis.train_quantile import fetch_basic_attributes


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False


def infer(N,
          P,
          hidden_dim,
          tau,
          date_array,
          stock_list,
          graph_model_path,
          nar_model_path,
          tensor_path,
          ret_col=7,
          graph=False,
          weighted_adj=False):
    T = date_array.shape[0]
    K_df = pd.read_csv(f'{graph_model_path}/{tau}/K_result.csv')
    K = K_df.loc[K_df['best_score'].argmin(), 'K']
    network = SGA(individual_num=N,
                  feature_dim=P,
                  K=K,
                  hidden_dim=hidden_dim,
                  num_layers=2)
    network.to(network.device)
    network.load_state_dict(torch.load(f'{graph_model_path}/{tau}/{K}/network_best.pth',
                                       map_location=network.device,
                                       weights_only=True))
    network.eval()

    nar = NAR(feature_dim=P).to(network.device)
    nar.load_state_dict(torch.load(f'{nar_model_path}/{tau}/network_best.pth',
                                   map_location=network.device,
                                   weights_only=True))
    nar.eval()

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
            adj_mat = network.encode_graph(X)
            r_prev = X[:, -1, ret_col]
            network_output = nar(X[:, -1, :], adj_mat, r_prev)
            result_dict[date][tau] = network_output.cpu().numpy()
            try:
                label_tensor = np.load(f'{tensor_path}/{date}/label.npy')
                result_dict[date]['ground_truth'] = label_tensor[:, 0]
            except FileNotFoundError:
                result_dict[date]['ground_truth'] = np.nan
            if graph:
                if weighted_adj:
                    mat_dict[date] = network.encode_weighted_adj(X).cpu().numpy()
                else:
                    mat_dict[date] = adj_mat.cpu().numpy()

    result_df = pd.concat(result_dict.values(), axis=0)
    return result_df, mat_dict, K


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--lag', type=int, default=lag,
                        help='Number of lagged value of each feature')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--ret-col', type=int, default=7,
                        help='Feature column index of lagged return (log_ret).')
    parser.add_argument('--cuda', type=int, default=0,
                        help='No. of GPU device.')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    hidden_dim = args.hidden

    graph_model_path = f'{project_path}/quantile_model/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_{horizon}'
    nar_model_path = f'{project_path}/nar_quantile_model/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_{horizon}'
    tensor_path = f'{project_path}/tensor/lag_{args.lag}_horizon_{horizon}'
    moment_path = f'{project_path}/nar_moment/hidden_{hidden_dim}_lr_{args.lr}_lag_{args.lag}_horizon_{horizon}'
    N, P, _ = fetch_basic_attributes(tensor_path, start_time, end_time)
    tolerance = 20
    size = 0.01

    os.makedirs(moment_path, exist_ok=True)

    tau_list = os.listdir(nar_model_path)
    tau_list.sort()
    tau_list = [float(x) for x in tau_list if isnumber(x)]

    date_list = os.listdir(tensor_path)
    date_list.sort()
    date_array = np.array(date_list)
    date_array = date_array[(date_array >= start_time) & (date_array <= end_time)]

    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)

    inference_dict = dict.fromkeys(tau_list, 0)
    for tau in tqdm(tau_list, desc='inference'):
        result_df, mat_dict, K = infer(N=N,
                                       P=P,
                                       hidden_dim=hidden_dim,
                                       tau=tau,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       graph_model_path=graph_model_path,
                                       nar_model_path=nar_model_path,
                                       tensor_path=tensor_path,
                                       ret_col=args.ret_col,
                                       graph=False)
        inference_dict[tau] = result_df

    feasible_stock_list = []
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
            feasible_stock_list.append(stock_name)
        except Exception as e:
            print(f'{stock_name} {e}')

    np.save(f'{project_path}/nar_feasible_stock_list.npy', np.array(feasible_stock_list))
