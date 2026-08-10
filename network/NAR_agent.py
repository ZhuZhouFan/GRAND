import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import sys
sys.path.append('.')
from network.NAR import NAR
from network.dataset import SGA_Dataset

class NAR_agent(object):
    def __init__(self,
                 individual_num: int,
                 feature_dim: int,
                 graph_model,
                 ret_col: int,
                 log_dir: str = None,
                 patience: int = 10,
                 batch_size: int = 1,
                 seed: int = 0,
                 learning_rate: float = 1e-5,
                 cuda: bool = True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)

        self.device = torch.device("cuda" if cuda & torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.individual_num = individual_num
        self.patience = patience
        self.ret_col = ret_col
        
        self.graph_model = graph_model.to(self.device)
        self.graph_model.eval()
        
        for param in self.graph_model.parameters():
            param.requires_grad = False
        
        self.network = NAR(feature_dim, cuda).to(self.device)
        self.optim = Adam(self.network.parameters(), lr=learning_rate)
        self.network.train()

        if log_dir is not None:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_score = 1e20
        self.early_stopping_count = 0

    def load_data(self, data_path, start_time, end_time, num_workers=10):
        dataset = SGA_Dataset(data_path, start_time, end_time)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
    def qr_loss(self, network_output, y, tau):
        resi = torch.sub(y, network_output)
        qr_vector = (F.relu(resi) * tau + F.relu(-1 * resi) * (1 - tau))
        return qr_vector.sum() / self.individual_num

    def train(self,
              epoch: int = 500,
              lambda_: float = 0,
              mse_loss: bool = False,
              tau: float = 0.5):
        
        for i in range(epoch):
            start_time = time.time()
            train_losses = list()
            coverage_ratio_list = list()
            
            for j, (X, y) in enumerate(self.data_loader):
                X = X[0, :, :, :].to(self.device)
                with torch.no_grad():
                    adj_mat = self.graph_model.encode_graph(X)
                feature = X[:, -1, :]
                r_prev = X[:, -1, self.ret_col]
                y = y[0, :, 0].to(self.device)
                
                network_output = self.network(feature, adj_mat, r_prev)
                
                L1_loss = 0
                for param in self.network.parameters():
                    L1_loss += torch.sum(torch.abs(param))

                if mse_loss:
                    loss = F.mse_loss(network_output, y) + lambda_ * L1_loss
                else:
                    loss = self.qr_loss(network_output, y, tau) + lambda_ * L1_loss

                coverage_ratio = (y <= network_output).sum() / self.individual_num
                coverage_ratio_list.append(coverage_ratio.item())
                train_losses.append(loss.item())

                self.update_params(self.optim, loss, networks=[self.network], retain_graph=False)

            end_time = time.time()
            dtime = end_time - start_time
            train_loss = torch.tensor(train_losses).mean()
            coverage_ratio = torch.tensor(coverage_ratio_list).mean()
            
            print(f'Epoch {i}/{epoch} Training Loss {train_loss:.6f} Time Consume {dtime:.3f}')
            self.writer.add_scalar('train/loss', train_loss, i)
            self.writer.add_scalar('train/cr', coverage_ratio.item(), i)

            if i % 3 == 0:
                self.evaluate(i, mse_loss, tau)
                torch.save(self.network.state_dict(), os.path.join(self.log_dir, 'network_final.pth'))

            if self.early_stopping_count >= self.patience:
                break

    def evaluate(self, i, mse_loss, tau):
        self.network.eval()
        with torch.no_grad():
            
            eval_losses = list()
            coverage_ratio_list = list()
            
            for j, (X, y) in enumerate(self.data_loader):
                X = X[0, :, :, :].to(self.device)
                adj_mat = self.graph_model.encode_graph(X)
                feature = X[:, -1, :]
                r_prev = X[:, -1, self.ret_col]
                y = y[0, :, 0].to(self.device)
                
                network_output = self.network(feature, adj_mat, r_prev)

                if mse_loss:
                    loss = F.mse_loss(network_output, y)
                else:
                    loss = self.qr_loss(network_output, y, tau)

                coverage_ratio = (y <= network_output).sum() / self.individual_num
                coverage_ratio_list.append(coverage_ratio.item())
                eval_losses.append(loss.item())

            eval_loss = torch.tensor(eval_losses).mean()
            coverage_ratio = torch.tensor(coverage_ratio_list).mean()
            
        self.writer.add_scalar('eval/loss', eval_loss.item(), i)
        self.writer.add_scalar('eval/cr', coverage_ratio.item(), i)
        print("-"*60)
        print(f'Evaluation {i} Loss (on training data) {eval_loss.item():.6f}')
        
        if eval_loss.item() < self.best_score:
            print('update model')
            self.best_score = eval_loss.item()
            self.best_cr = coverage_ratio.item()
            self.writer.add_scalar('best/cr', coverage_ratio.item(), i)
            
            torch.save(self.network.state_dict(), os.path.join(self.log_dir, 'network_best.pth'))
            self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1
        print("-"*60)
        self.network.train()

    def update_params(self, optim, loss, networks, retain_graph=False, grad_cliping=None):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if grad_cliping:
            for net in networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        optim.step() 