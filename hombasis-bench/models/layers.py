import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GINConv, GCNConv, GATConv


"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False
            
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
            
        # note: batchnorm included in default MLP
        self.conv = GINConv(MLP(in_channels=in_dim, hidden_channels=out_dim, out_channels=out_dim, num_layers=2), train_eps=True)

    def forward(self, x, edge_index):
        h_in = x # for residual connection
        
        h = self.conv(x=x, edge_index=edge_index)
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        return h


class GINLayerSig(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False
            
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
            
        # note: batchnorm included in default MLP
        self.conv = GINConv(MLP(in_channels=in_dim, hidden_channels=out_dim, out_channels=out_dim, num_layers=2, act="sigmoid"), train_eps=True)

    def forward(self, x, edge_index):
        h_in = x # for residual connection
        
        h = self.conv(x=x, edge_index=edge_index)
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        return h


"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_dim != out_dim:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        
    def forward(self, x, edge_index):
        h_in = x   # to be used for residual connection
        
        h = self.conv(x=x, edge_index=edge_index)
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        h = F.relu(h) # non-linear activation

        if self.residual:
            h = h_in + h # residual connection

        return h


"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, batch_norm, residual):
        super().__init__()
        self.residual = residual
        self.batch_norm = batch_norm
            
        if in_dim != (out_dim*num_heads):
            self.residual = False
        
        # default add self loops is True
        self.gatconv = GATConv(in_dim, out_dim, num_heads, add_self_loops=False)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, x, edge_index):
        h_in = x # for residual connection

        h = self.gatconv(x=x, edge_index=edge_index)
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        h = F.relu(h) # non-linear activation
            
        if self.residual:
            h = h_in + h # residual connection

        return h
    
    
"""
    MLP Layer used after graph vector representation
"""
class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y