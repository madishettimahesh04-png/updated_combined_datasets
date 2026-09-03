import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv,global_mean_pool

class GNN(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.gat1=GATConv(in_channels=9,out_channels=hidden_dim,heads=heads,concat=True,dropout=dropout)
        self.bn1=nn.BatchNorm1d(hidden_dim*heads)
        self.sage1=SAGEConv(hidden_dim*heads,hidden_dim)
        self.bn2=nn.BatchNorm1d(hidden_dim)
        self.sage2=SAGEConv(hidden_dim,hidden_dim)
        self.bn3=nn.BatchNorm1d(hidden_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,data):
        x=data.x.float(); edge_index=data.edge_index; batch=data.batch
        x=self.dropout(F.elu(self.bn1(self.gat1(x,edge_index))))
        x=self.dropout(F.relu(self.bn2(self.sage1(x,edge_index))))
        r=x
        x=self.dropout(F.relu(self.bn3(self.sage2(x,edge_index))+r))
        return global_mean_pool(x,batch)

class Model(nn.Module):
    def __init__(self,desc_dim,hidden_dim,heads,dropout,desc_hidden,mlp_hidden):
        super().__init__()
        self.gnn=GNN(hidden_dim,heads,dropout)
        self.desc_net=nn.Sequential(
            nn.Linear(desc_dim,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(128,64),nn.ReLU())
        fusion_dim=hidden_dim*4+64
        self.fc=nn.Sequential(
            nn.Linear(fusion_dim,mlp_hidden),nn.BatchNorm1d(mlp_hidden),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(mlp_hidden,mlp_hidden//2),nn.BatchNorm1d(mlp_hidden//2),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,g1,g2,descriptors):
        g1e=self.gnn(g1); g2e=self.gnn(g2)
        gf=torch.cat([g1e,g2e,torch.abs(g1e-g2e),g1e*g2e],dim=1)
        df=self.desc_net(descriptors)
        return self.fc(torch.cat([gf,df],dim=1)).squeeze(-1)
