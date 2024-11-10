import torch
import torch.nn as nn
import torch.nn.functional as F


class SGA(nn.Module):
    def __init__(self, 
                 individual_num:int, 
                 feature_dim:int,
                 K:int,
                 hidden_dim:int = 128,
                 num_layers:int = 2,
                 dropout:float = 0.00,
                 cuda:bool = True):
        super().__init__()
        self.device = torch.device(
            "cuda" if cuda & torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.individual_num = individual_num
        self.dropout = nn.Dropout(dropout)
        
        self.rnn = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            ) # [N, S, feature_dim] -> [N, 1, hidden_dim]

        self.hidden_concept_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hs = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hs_fore = nn.Linear(hidden_dim, hidden_dim)

        self.fc_hs_back = nn.Linear(hidden_dim, hidden_dim)
        self.fc_indi = nn.Linear(hidden_dim, hidden_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax_t2s = nn.Softmax(dim = 0)
        
        self.fc_out_hs = nn.Linear(hidden_dim, 1)
        self.fc_out_indi = nn.Linear(hidden_dim, 1)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.K = K
        self.mask_row = torch.linspace(0, individual_num - 1, individual_num).reshape([-1, 1]).repeat(1, K).reshape(1, -1).long().to(self.device)
        self.off_diagonal_mat = (torch.ones(individual_num, individual_num) - torch.eye(individual_num)).to(self.device)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data) 
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def cal_cos_similarity(self, x, y):
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity
    
    def encoder_graph(self, x):
        gru_output, _ = self.rnn(x)
        gru_output = gru_output[:, -1, :]

        h_shared_info = gru_output
        similarity_mat = self.cal_cos_similarity(h_shared_info, h_shared_info)
        similarity_mat = similarity_mat * self.off_diagonal_mat

        mask_column = torch.topk(similarity_mat, self.K, dim = 1)[1].reshape(1, -1) 
        mask = torch.zeros([self.individual_num, self.individual_num], device = x.device)
        mask[self.mask_row, mask_column] = 1
        topK_similarity_mat = similarity_mat * mask
        topK_similarity_mat = topK_similarity_mat[:, topK_similarity_mat.sum(0) != 0]
        
        adj_mat = torch.zeros([self.individual_num, self.individual_num], device = x.device)
        for concept_idx in range(topK_similarity_mat.shape[1]):
            non_zero_idx = topK_similarity_mat[:, concept_idx]
            connected_stocks = torch.where(non_zero_idx)[0]
            for j, sender in enumerate(connected_stocks[:-1]):
                receiver = connected_stocks[j+1]
                adj_mat[sender, receiver] = 1
                adj_mat[receiver, sender] = 1
        return adj_mat
    
    def forward(self, x):
        gru_output, _ = self.rnn(x)
        gru_output = gru_output[:, -1, :]
        
        h_shared_info = gru_output
        similarity_mat = self.cal_cos_similarity(h_shared_info, h_shared_info)
        diag = similarity_mat.diagonal(0)
        similarity_mat = similarity_mat * self.off_diagonal_mat
        
        mask_column = torch.topk(torch.abs(similarity_mat), self.K, dim = 1)[1].reshape(1, -1) 
        mask = torch.zeros([self.individual_num, self.individual_num], device = x.device)
        mask[self.mask_row, mask_column] = 1
        topK_similarity_mat = similarity_mat * mask 
        topK_similarity_mat = topK_similarity_mat + torch.diag_embed((topK_similarity_mat.sum(0)!=0).float() * diag)
        
        concept_feature = torch.t(h_shared_info).mm(topK_similarity_mat).t()
        concept_feature = concept_feature[concept_feature.sum(1)!=0]
        concept_feature = self.leaky_relu(self.hidden_concept_fc(concept_feature))
        concept_feature = self.dropout(concept_feature)
        
        concept_similarity_mat = self.cal_cos_similarity(h_shared_info, concept_feature)
        concept_attention_mat = self.softmax_t2s(concept_similarity_mat)
        
        h_shared_info = concept_attention_mat.mm(concept_feature) 
        h_shared_info = self.leaky_relu(self.fc_hs(h_shared_info))
        h_shared_info = self.dropout(h_shared_info)

        h_shared_back = self.leaky_relu(self.fc_hs_back(h_shared_info))
        output_hs = self.leaky_relu(self.fc_hs_fore(h_shared_info))
        output_hs = self.dropout(output_hs)

        individual_info  = gru_output - h_shared_back
        output_indi = self.leaky_relu(self.fc_indi(individual_info))
        output_indi = self.dropout(output_indi)
       
        all_info = output_hs + output_indi
        pred_all = self.fc_out(all_info).squeeze()
        return pred_all
    
class MLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data) 
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x