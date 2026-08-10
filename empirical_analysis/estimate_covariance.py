"""
Estimate sparse conditional covariance matrices via GRAND node fusion and QCM.

Inputs: trained quantile / sigma / MLP models, QCM moment CSVs, and feature
tensors; CLI flags ``--lr``, ``--hidden``, ``--lag``, ``--cuda``, and
``--graph`` (``variance`` | ``mean`` | ``tail`` | ``median``).
Operations: extract a sparse graph, estimate off-diagonal covariances for
connected pairs by fusing nodes and re-inferring quantiles, fill diagonals from
QCM variances, and project correlations to the nearest PD matrix (NCM).
Outputs: ``{project_path}/{graph}_graph_cov/hidden_*_.../{date}_cov.npy``.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, end_time, horizon, lag
from network.model import SGA, MLP
from network.NCM import nearcorr, cor2cov, cov2cor
from network.QCM import QCM_regression
from empirical_analysis.train_quantile import fetch_basic_attributes
from empirical_analysis.moment_inference import infer


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def estimate_covariance(sender,
                        receiver,
                        feature,
                        tau_list,
                        hidden_dim,
                        quantile_model_path,
                        mlp_model_path,
                        diagonal_elements,
                        device_id=0,
                        device='cuda'):
    
    torch.cuda.set_device(device_id)
    quantile_df = pd.DataFrame(columns=tau_list + ['ground_truth'], index=[0])
    N, _, P = feature.shape

    selected_features = np.concatenate([feature[sender:sender+1, :, :], feature[receiver:receiver+1, :, :]], axis=-1)
    selected_features = torch.Tensor(selected_features)
    other_vertices = np.delete(feature, [sender, receiver], axis=0)

    for tau in tau_list:
        node_fuse_model = MLP(2 * P, hidden_dim, P).to(device)
        
        node_fuse_model.load_state_dict(torch.load(f'{mlp_model_path}/{tau}/mlp_best.pth',
                                        map_location=device))
        with torch.no_grad():
            fused_vertex = node_fuse_model.forward(selected_features.to(device))
            fused_feature = torch.cat([fused_vertex,
                                       torch.tensor(other_vertices).float().to(fused_vertex.device)],
                                      axis=0)

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
            network_output = network.forward(torch.Tensor(fused_feature).to(device))
            
        quantile_df.loc[0, tau] = network_output[0].cpu().detach().numpy()
    quantile_df.loc[0, 'ground_truth'] = np.nan
    quantile_df = quantile_df.astype('float32')
    QCM_table = QCM_regression(quantile_df)
    covariance = (QCM_table['variance'] - diagonal_elements[sender, sender] - diagonal_elements[receiver, receiver])/2
    return covariance.values.item()


def estimate_covariance_mat(adj_mat, 
                            diagonal_elements,
                            feature, 
                            hidden_dim,
                            tau_list, 
                            quantile_model_path,
                            mlp_model_path,
                            n_jobs=5,
                            device_id=0):
    covariance_mat = np.zeros_like(adj_mat)
    lower_mat = np.tril(adj_mat)
    senders, receivers = np.where(lower_mat)
    senders_receivers = zip(senders, receivers)

    # for sender, receiver in senders_receivers:
    #     a = estimate_covariance(sender, receiver, feature, tau_list,
    #                                            hidden_dim, quantile_model_path,
    #                                            mlp_model_path, diagonal_elements,
    #                                            device_id)
    
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
    parser.add_argument('--lag', type=int, default=lag,
                        help='Number of lagged value of each feature')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--graph', type=str, default='variance', choices=['variance', 'mean', 'tail', 'median'],
                        help='Select the sparse structure')
    args = parser.parse_args()

    device_id = args.cuda
    torch.cuda.set_device(device_id)
    device = 'cuda'
    lag_order = args.lag
    model_label = f'hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_{horizon}'
    quantile_model_path = f'{project_path}/quantile_model/{model_label}'
    sigma_model_path = f'{project_path}/sigma_model/{model_label}'
    quantile_tensor_path = f'{project_path}/tensor/lag_{lag_order}_horizon_{horizon}'
    sigma_tensor_path = f'{project_path}/sigma_tensor/{model_label}'
    moment_path = f'{project_path}/moment/{model_label}'
    mlp_model_path = f'{project_path}/mlp_model/{model_label}'
    
    hidden_dim = args.hidden
    N, P, _ = fetch_basic_attributes(quantile_tensor_path, valid_time, end_time)
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)
    index_df = pd.read_csv(f'{project_path}/kline_week_index/000001.XSHG.csv')
    date_array = index_df.loc[(index_df['date'] >= valid_time) & (index_df['date'] <= end_time), 'date'].values

    if args.graph == 'variance':
        result_df, mat_dict, K = infer(N=N,
                                       P=P+4,
                                       hidden_dim=hidden_dim,
                                       tau=0.0,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=sigma_model_path,
                                       tensor_path=sigma_tensor_path,
                                       graph=True)
        cov_path = f'{project_path}/sigma_graph_cov/{model_label}'    
    elif args.graph == 'mean':
        result_df, mat_dict, K = infer(N=N,
                                       P=P,
                                       hidden_dim=hidden_dim,
                                       tau=0.0,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=quantile_model_path,
                                       tensor_path=quantile_tensor_path,
                                       graph=True)
        cov_path = f'{project_path}/mean_graph_cov/{model_label}'
    elif args.graph == 'tail':
        result_df, mat_dict, K = infer(N=N,
                                       P=P,
                                       hidden_dim=hidden_dim,
                                       tau=0.05,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=quantile_model_path,
                                       tensor_path=quantile_tensor_path,
                                       graph=True)
        cov_path = f'{project_path}/tail_graph_cov/{model_label}'
    elif args.graph == 'median':
        result_df, mat_dict, K = infer(N=N,
                                       P=P,
                                       hidden_dim=hidden_dim,
                                       tau=0.5,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=quantile_model_path,
                                       tensor_path=quantile_tensor_path,
                                       graph=True)
        cov_path = f'{project_path}/median_graph_cov/{model_label}'
    else:
        raise ValueError(f'Invalid graph type: {args.graph}')
        
    os.makedirs(cov_path, exist_ok=True)
    
    moment_exist_list = os.listdir(moment_path)
    moment_exist_list = [x[:-4] for x in moment_exist_list]
    moment_exist_list.sort()
    
    tau_list = os.listdir(quantile_model_path)
    tau_list.sort()
    tau_list = [float(x) for x in tau_list if isnumber(x)]

    for date in tqdm(date_array):

        if os.path.exists(f'{cov_path}/{date}_cov.npy'):
            continue

        adj_mat = mat_dict[date]

        # with a determined order
        feasible_stock_list = np.intersect1d(moment_exist_list, stock_list).tolist()  # may lose the order
        feasible_stock_list.sort()  # ensure the order of stocks

        remain_index = list(map(lambda x: x in feasible_stock_list, stock_list))
        adj_mat = adj_mat[remain_index][:, remain_index]
        covariance_mat = np.zeros_like(adj_mat)
        # diagnoal elements
        diagonal_elements = np.eye(adj_mat.shape[0])
        for stock_indice, stock_name in enumerate(feasible_stock_list):
            tem = pd.read_csv(f'{moment_path}/{stock_name}.csv', index_col='date')
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
        correlation_mat = cov2cor(covariance_mat)
        try:
            pd_correlation_mat = nearcorr(correlation_mat, max_iterations=1000)
            pd_covariance_mat = cor2cov(pd_correlation_mat, np.diagonal(covariance_mat))
            np.save(f'{cov_path}/{date}_cov.npy', pd_covariance_mat)
        except Exception as e:
            print(e)