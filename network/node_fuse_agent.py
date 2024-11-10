import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch.nn.functional as F
from dataset import SGA_Dataset
from torch.utils.data import DataLoader

from model import SGA, MLP


class mlp_agent(object):

    def __init__(self,
                 individual_num: int,
                 feature_dim: int,
                 hidden_dim: int,
                 K: int,
                 dropout: float = 0.0,
                 log_dir: str = None,
                 num_layers: int = 2,
                 patience: int = 30,
                 batch_size: int = 32,
                 seed: int = 0,
                 learning_rate: float = 1e-5,
                 cuda: bool = True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)

        self.device = torch.device(
            "cuda" if cuda & torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.individual_num = individual_num
        self.patience = patience

        self.SGA = SGA(individual_num=individual_num - 1,
                       feature_dim=feature_dim,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       dropout=dropout,
                       K=K,
                       cuda=cuda
                       ).to(self.device)
        self.mlp = MLP(feature_dim=2 * feature_dim,
                       hidden_dim=hidden_dim,
                       output_dim=feature_dim).to(self.device)

        self.optim = Adam(self.mlp.parameters(), lr=learning_rate)

        self.mlp.train()

        if log_dir is not None:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_score = 1e20
        self.early_stopping_count = 0

    def load_data(self,
                  data_path,
                  start_time,
                  valid_time,
                  end_time,
                  num_workers=10):

        dataset_train = SGA_Dataset(data_path, start_time, valid_time)
        self.train_loader = DataLoader(
            dataset_train, batch_size=1, shuffle=True, num_workers=num_workers)
        dataset_test = SGA_Dataset(data_path, valid_time, end_time)
        self.test_loader = DataLoader(
            dataset_test, batch_size=1, shuffle=True, num_workers=num_workers)

    def fuse_node(self, cross_feature, cross_label):
        
        cross_feature = cross_feature[0, :, :, :]
        cross_label = cross_label[0, :, :]
        a, b = np.random.choice(cross_feature.shape[0] - 1, 2, replace=False)

        selected_features = np.concatenate([cross_feature[a:a+1, :, :], cross_feature[b:b+1, :, :]], axis=-1)
        new_vertex = self.mlp.forward(
            torch.Tensor(selected_features).to(self.device))
        other_vertices = torch.Tensor(
            np.delete(cross_feature, [a, b], axis=0)).to(self.device)
        feature = torch.cat([new_vertex, other_vertices], axis=0)

        new_label = cross_label[a, :] + cross_label[b, :]
        other_labels = np.delete(cross_label, [a, b], axis=0)
        label = np.concatenate(
            [np.ones([1, 1]) * new_label, other_labels], axis=0)
        return feature, torch.Tensor(label[:, 0]).to(self.device)

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

            for j, (X, y) in enumerate(self.train_loader):
                batch_loss = 0

                for _ in range(self.batch_size):    
                    
                    fused_feature, fused_label = self.fuse_node(X.detach().numpy(), y.detach().numpy())
                    
                    network_output = self.SGA.forward(fused_feature)

                    L1_loss = 0
                    for param in self.mlp.parameters():
                        L1_loss += torch.sum(torch.abs(param))

                    if mse_loss:
                        loss = F.mse_loss(
                            network_output, fused_label) + lambda_ * L1_loss
                    else:
                        loss = self.qr_loss(
                            network_output, fused_label, tau) + lambda_ * L1_loss

                batch_loss += loss

                train_losses.append(loss.item())

                self.update_params(self.optim, batch_loss, networks=[
                                   self.mlp], retain_graph=False)

            # self.scheduler.step()
            end_time = time.time()
            dtime = end_time - start_time
            train_loss = torch.tensor(train_losses).mean()

            print(f'Epoch {i}/{epoch} Training Loss {train_loss:.6f} Time Consume {dtime:.3f}')
            self.writer.add_scalar('train/loss', train_loss, i)

            if i % 3 == 0:
                self.evaluate(i, mse_loss, tau)
                torch.save(self.mlp.state_dict(), os.path.join(
                    self.log_dir, 'mlp_final.pth'))

            if self.early_stopping_count >= self.patience:
                break

    def evaluate(self,
                 i,
                 mse_loss,
                 tau):
        self.mlp.eval()
        self.SGA.eval()
        
        with torch.no_grad():
            test_losses = list()
            
            for j, (X, y) in enumerate(self.test_loader):
                batch_loss = 0
                for _ in range(self.batch_size):

                    fused_feature, fused_label = self.fuse_node(X.detach().numpy(), y.detach().numpy())
                    network_output = self.SGA.forward(fused_feature)

                    if mse_loss:
                        loss = F.mse_loss(
                            network_output, fused_label)
                    else:
                        loss = self.qr_loss(
                            network_output, fused_label, tau)

                batch_loss += loss

                test_losses.append(loss.item())

            test_loss = torch.tensor(test_losses).mean()
            
        self.writer.add_scalar('test/loss', test_loss.item(), i)
        print("-"*60)
        print(f'Evaluation {i} Loss {test_loss.item():.6f}')
        if test_loss.item() < self.best_score:
            print('update model')
            self.best_score = test_loss.item()

            torch.save(self.mlp.state_dict(), os.path.join(
                self.log_dir, 'mlp_best.pth'))
            self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1
        print("-"*60)
        self.mlp.train()
        self.SGA.train()

    def update_params(self, optim, loss, networks, retain_graph=False,
                      grad_cliping=None):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if grad_cliping:
            for net in networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        optim.step()
