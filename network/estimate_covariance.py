import argparse
import os
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from model import SGA, MLP
from NCM import nearcorr, cor2cov, cov2cor
from QCM import QCM_regression
from infer import infer
from tqdm import tqdm
from train import fetch_basic_attributes


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False


def obtain_valid_stock_list(kline_day_path, start_time, end_time,
                            null_patience=0.15):
    existing_stock_list = os.listdir(kline_day_path)
    existing_stock_list = [x[:-4] for x in existing_stock_list]
    existing_stock_list.sort()

    valid_stock_list = []
    for stock_name in existing_stock_list:
        df = pd.read_csv(
            f'{kline_day_path}/{stock_name}.csv', index_col='date')
        df.loc[(df['volume'] < 1) | (df['total_turnover'] < 1), :] = np.nan
        df = df.loc[start_time:end_time, :]
        if (df.isna().sum().max()) < (null_patience * df.shape[0]):
            valid_stock_list.append(stock_name)
    return np.array(valid_stock_list)

def estimate_covariance(sender, receiver,
                        feature, tau_list,
                        hidden_dim,
                        quantile_model_path,
                        mlp_model_path,
                        diagonal_elements,
                        device_id=0,
                        device='cuda'):
    torch.cuda.set_device(device_id)
    quantile_df = pd.DataFrame(columns=tau_list + ['ground_truth'], index=[0])
    N, P, S = feature.shape

    selected_features = np.concatenate(
        [feature[sender:sender+1, :, :], feature[receiver:receiver+1, :, :]], axis=-1)
    selected_features = torch.Tensor(selected_features)
    other_vertices = np.delete(feature, [sender, receiver], axis=0)

    for tau in tau_list:
        node_fuse_model = MLP(P, hidden_dim, P).to(device)
        node_fuse_model.load_state_dict(torch.load(f'{mlp_model_path}/{tau}/mlp_best.pth',
                                        map_location=device))
        with torch.no_grad():
            fused_vertex = node_fuse_model.forward(
                selected_features.to(device))
            fused_feature = torch.cat([fused_vertex, other_vertices], axis=0)

        K_df = pd.read_csv(f'{quantile_model_path}/{tau}/K_result.csv')
        K = K_df.loc[K_df['best_score'].argmin(), 'K']
        network = SGA(individual_num=N - 1,
                      feature_dim=P,
                      K=K,
                      hidden_dim=hidden_dim,
                      num_layers=2,
                      dropout=0.0).to(device)
        network.mask_row.to(device)
        network.off_diagonal_mat.to(device)
        network.load_state_dict(torch.load(f'{quantile_model_path}/{tau}/{K}/network_best.pth',
                                           map_location=device))
        network.eval()
        with torch.no_grad():
            network_output = network.forward(
                torch.Tensor(fused_feature).to(device))
        quantile_df.loc[0, tau] = network_output[0].cpu().detach().numpy()
    quantile_df.loc[0, 'ground_truth'] = np.nan
    quantile_df = quantile_df.astype('float32')
    QCM_table = QCM_regression(quantile_df)
    covariance = (QCM_table['variance'] - diagonal_elements[sender,
                  sender] - diagonal_elements[receiver, receiver])/2
    return covariance.values.item()


def estimate_covariance_mat(adj_mat, diagonal_elements,
                            feature, hidden_dim,
                            tau_list, quantile_model_path, mlp_model_path,
                            n_jobs=5, device_id=0):
    covariance_mat = np.zeros_like(adj_mat)
    lower_mat = np.tril(adj_mat)
    senders, receivers = np.where(lower_mat)
    senders_receivers = zip(senders, receivers)

    covariance_list = Parallel(n_jobs=n_jobs)(delayed(estimate_covariance)
                                              (sender, receiver, feature, tau_list,
                                               hidden_dim, quantile_model_path,
                                               mlp_model_path, diagonal_elements,
                                               device_id)
                                              for sender, receiver in senders_receivers)

    for sender_indice in range(len(senders)):
        sender = senders[sender_indice]
        receiver = receivers[sender_indice]
        covariance_mat[sender, receiver] = covariance_list[sender_indice]
    covariance_mat = covariance_mat + covariance_mat.transpose() + diagonal_elements
    return covariance_mat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device number')
    parser.add_argument('--start-time', type=str, default='2019-01-01',
                        help='The beginning of estimating covariance')
    parser.add_argument('--end-time', type=str, default='2023-09-01',
                        help='The ending of estimating covariance')
    parser.add_argument('--lag', type=int,
                        help='Number of lagged value of each feature')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--hidden', type=int,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--data-folder', type=str,
                        help='Path to your data folder')
    args = parser.parse_args()

    device_id = args.cuda
    torch.cuda.set_device(device_id)
    device = 'cuda'
    lag_order = args.lag
    data_path = args.data_folders
    model_label = f'hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1'
    quantile_model_path = f'{data_path}/quantile_model/{model_label}'
    sigma_model_path = f'{data_path}/sigma_model/{model_label}'
    quantile_tensor_path = f'{data_path}/tensor/lag_{lag_order}_horizon_1'
    sigma_tensor_path = f'{data_path}/sigma_tensor/{model_label}'
    moment_path = f'{data_path}/moment/{model_label}'
    mlp_model_path = f'{data_path}/mlp_model/{model_label}'
    
    start_time = args.start_time
    end_time = args.end_time

    hidden_dim = args.hidden
    N, P, _ = fetch_basic_attributes(quantile_tensor_path, start_time, end_time)

    cov_path = f'{data_path}/sigma_graph_cov/{model_label}'
    if not os.path.exists(cov_path):
        os.makedirs(cov_path)
    
    moment_exist_list = os.listdir(moment_path)
    moment_exist_list = [x[:-4] for x in moment_exist_list]
    moment_exist_list.sort()
    tau_list = os.listdir(quantile_model_path)
    tau_list.sort()
    tau_list = [float(x) for x in tau_list if isnumber(x)]

    index_df = pd.read_csv(f'{data_path}/kline_week_index/000001.XSHG.csv')
    date_array = index_df.loc[(index_df['date'] >= start_time) & (index_df['date'] <= end_time), 'date'].values

    stock_list = obtain_valid_stock_list(f'{data_path}/kline_day', '2010-01-01', '2018-12-31')

    result_df, mat_dict, K = infer(N=N,
                                   P=P+4,
                                   hidden_dim=hidden_dim,
                                   tau=0.0,
                                   date_array=date_array,
                                   stock_list=stock_list,
                                   model_path=sigma_model_path,
                                   tensor_path=sigma_tensor_path,
                                   device='cuda',
                                   graph=True)

    for date in tqdm(date_array):

        if os.path.exists(f'{cov_path}/{date}_pd_cov.npy'):
            continue

        adj_mat = mat_dict[date]

        # with a determined order
        current_index_stock_list = stock_list
        possible_stock_list = np.intersect1d(moment_exist_list, current_index_stock_list).tolist()  # may lose the order
        possible_stock_list.sort()  # ensure the order of stocks

        remain_index = list(
            map(lambda x: x in possible_stock_list, current_index_stock_list))
        adj_mat = adj_mat[remain_index][:, remain_index]

        covariance_mat = np.zeros_like(adj_mat)
        # diagnoal elements
        diagonal_elements = np.eye(adj_mat.shape[0])
        for stock_indice, stock_name in enumerate(possible_stock_list):
            tem = pd.read_csv(
                f'{moment_path}/{stock_name}.csv', index_col='date')
            variance = tem.loc[date, 'variance']
            diagonal_elements[stock_indice, stock_indice] = variance
        # off-diagonal elements
        lower_mat = np.tril(adj_mat)
        senders, receivers = np.where(lower_mat)

        if senders.shape[0] == 0:
            print('Warning: No off-diagonal elements!!!')
            continue

        feature = np.load(f'{quantile_tensor_path}/{date}/feature.npy')
        covariance_mat = estimate_covariance_mat(adj_mat,
                                                 diagonal_elements,
                                                 feature,
                                                 hidden_dim,
                                                 tau_list,
                                                 quantile_model_path,
                                                 mlp_model_path,
                                                 n_jobs=5,
                                                 device_id=device_id)
        np.save(f'{cov_path}/{date}_cov.npy', covariance_mat)
        correlation_mat = cov2cor(covariance_mat)
        try:
            pd_correlation_mat = nearcorr(correlation_mat, max_iterations=1e3)
            pd_covariance_mat = cor2cov(
                pd_correlation_mat, np.diagonal(covariance_mat))
            np.save(f'{cov_path}/{date}_pd_cov.npy', pd_covariance_mat)
            np.save(f'{cov_path}/{date}_stocks.npy',
                    np.array(possible_stock_list))
        except Exception as e:
            print(e)
