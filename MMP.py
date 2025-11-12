import torch
import torch.nn as nn 

class MaxLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features)*0.1)

    def forward(self, x):
        out = self.W.unsqueeze(0) + x.unsqueeze(1)  # shape: (batch, out_features, in_features)
        out = torch.max(out, dim=2)[0]
        return out

class MinLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features)*0.1)

    def forward(self, x):
        out = self.W.unsqueeze(0) + x.unsqueeze(1)  # shape: (batch, out_features, in_features)
        out = torch.min(out, dim=2)[0]
        return out
    
    def restricted_normalize(self,D,f_func,g_func):
        with torch.no_grad():
            g_vals = g_func(D)
            f_vals = f_func(D)
            g_exp = g_vals.unsqueeze(2) # shape:(N,out_features)
            f_exp = f_vals.unsqueeze(1) # shape:(N,in_features)
            nu_D = (g_exp - f_exp).max(dim=0)[0]# (out_features,in_features)

            self.W.data.copy_(nu_D)#重みを更新
    
class MaxMinQNet(nn.Module):#nn.Module:PyTorch が提供してくれる基本クラス
    def __init__(self, obs_dim, n_actions,hidden_dim=64): #abs_dim:状態の次元
        super().__init__()
        self.fc_in = nn.Linear(obs_dim,hidden_dim)
        self.min1 = MinLinear(hidden_dim,hidden_dim) 
        self.max1 = MaxLinear(hidden_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,hidden_dim)
        self.fc_out = nn.Linear(hidden_dim,n_actions)

    def forward(self,x):
        x = self.fc_in(x)
        x = self.max1(x)
        x = self.min1(x)
        #x = self.fc(x)
        #x = self.max1(x)
        q = self.fc_out(x)
        return q