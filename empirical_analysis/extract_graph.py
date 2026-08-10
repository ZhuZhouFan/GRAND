"""
Extract out-of-sample sparse adjacency graphs from trained SGA models.

Inputs: trained quantile and sigma checkpoints / tensors under
``{project_path}/quantile_model|sigma_model|tensor|sigma_tensor/...``;
``valid_stocks.npy``. CLI: ``--lr``, ``--hidden``, ``--lag``, ``--cuda``,
optional ``--weighted``.
Operations: for OOS dates, encode binary (or signed) graphs at the left-tail
quantile (``tau=0.05``), median (``0.5``), conditional mean (``0.0``), and
sigma-model variance graph.
Outputs: ``{project_path}/{Q,M,E,V}_graph.npy`` (or ``*_wgraph.npy`` if
``--weighted``), each a date-keyed dict of adjacency matrices used by the
downstream causal / connectedness analyses.
"""

import torch
import os
import numpy as np
import argparse

import sys
sys.path.append('.')
from empirical_analysis.moment_inference import infer
from empirical_analysis.train_quantile import fetch_basic_attributes
from config import project_path, start_time, valid_time, end_time, horizon, lag

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--lag', type=int, default=lag,
                        help='Number of lagged value of each feature')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--weighted', action='store_true', default=False,
                        help='whether extracted graph with sign')

    parser.add_argument('--cuda', type=int, default=0,
                        help='No. of GPU device.')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    model_label = f'hidden_{args.hidden}_lr_{args.lr}_lag_{args.lag}_horizon_{horizon}'
    quantile_model_path = f'{project_path}/quantile_model/{model_label}'
    quantile_tensor_path = f'{project_path}/tensor/lag_{args.lag}_horizon_{horizon}'
    sigma_tensor_path = f'{project_path}/sigma_tensor/{model_label}'
    sigma_model_path = f'{project_path}/sigma_model/{model_label}'
    N, P, _ = fetch_basic_attributes(quantile_tensor_path, start_time, end_time)

    date_list = os.listdir(quantile_tensor_path)
    date_list.sort()
    date_array = np.array(date_list)
    date_array = date_array[(date_array >= valid_time) & (date_array <= end_time)]

    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)

    # Q_graph (left-tail VaR)
    _, Q_adjs, _ = infer(N=N,
                         P=P,
                         hidden_dim=args.hidden,
                         tau=0.05,
                         date_array=date_array,
                         stock_list=stock_list,
                         model_path=quantile_model_path,
                         tensor_path=quantile_tensor_path,
                         graph=True,
                         weighted_adj=args.weighted)
    if args.weighted:
        np.save(f'{project_path}/Q_wgraph.npy', Q_adjs)
    else:
        np.save(f'{project_path}/Q_graph.npy', Q_adjs)

    _, M_adjs, _ = infer(N=N,
                         P=P,
                         hidden_dim=args.hidden,
                         tau=0.5,
                         date_array=date_array,
                         stock_list=stock_list,
                         model_path=quantile_model_path,
                         tensor_path=quantile_tensor_path,
                         graph=True,
                         weighted_adj=args.weighted)
    if args.weighted:
        np.save(f'{project_path}/M_wgraph.npy', M_adjs)
    else:
        np.save(f'{project_path}/M_graph.npy', M_adjs)

    # E_graph (conditional mean)
    _, E_adjs, _ = infer(N=N,
                         P=P,
                         hidden_dim=args.hidden,
                         tau=0.0,
                         date_array=date_array,
                         stock_list=stock_list,
                         model_path=quantile_model_path,
                         tensor_path=quantile_tensor_path,
                         graph=True,
                         weighted_adj=args.weighted)
    if args.weighted:
        np.save(f'{project_path}/E_wgraph.npy', E_adjs)
    else:
        np.save(f'{project_path}/E_graph.npy', E_adjs)

    # V_graph (conditional variance / sigma model)
    _, V_adjs, _ = infer(N=N,
                         P=P + 4,
                         hidden_dim=args.hidden,
                         tau=0.0,
                         date_array=date_array,
                         stock_list=stock_list,
                         model_path=sigma_model_path,
                         tensor_path=sigma_tensor_path,
                         graph=True,
                         weighted_adj=args.weighted)
    if args.weighted:
        np.save(f'{project_path}/V_wgraph.npy', V_adjs)
    else:
        np.save(f'{project_path}/V_graph.npy', V_adjs)
