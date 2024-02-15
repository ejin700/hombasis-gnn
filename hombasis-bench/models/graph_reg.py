import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import scatter
from models.layers import GCNLayer, GATLayer, GINLayer, MLPReadout, GINLayerSig


class GINGraphReg(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            count_dim: int,
            # pe_dim: int,
            num_layers: int,
            batch_norm: bool,
            residual: bool,
            readout: str,
        ):
        
        super(GINGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        # self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout

        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = hidden_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
            concat_feature_dim = hidden_dim + count_dim
        else:
            concat_feature_dim = hidden_dim
        
        # GIN message passing layers
        self.convs = nn.ModuleList([GINLayer(hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual) for _ in range(self.num_layers-1)])
        self.convs.insert(0, GINLayer(concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual))

        # decoder
        self.decoder = MLPReadout(hidden_dim, 1)

        
    def forward(self, x, edge_index, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
            
        else:
            h = atom_h
            
        for layer in self.convs:
            h = layer(x=h, edge_index=edge_index)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h)
        
        return h

    
class GCNGraphReg(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            count_dim: int,
            num_layers: int,
            batch_norm: bool,
            residual: bool,
            readout: str,
        ):
        
        super(GCNGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout

        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = hidden_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
        
        concat_feature_dim = hidden_dim + count_dim
        
        # GCN message passing layers        
        self.convs = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual) for _ in range(self.num_layers-1)])
        self.convs.insert(0, GCNLayer(concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual))
        
        # decoder
        self.decoder = MLPReadout(hidden_dim, 1)
        
    def forward(self, x, edge_index, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
        else:
            h = atom_h

        # model step
        for conv in self.convs:
            h = conv(x=h, edge_index=edge_index)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h)
        
        return h
    

class GATGraphReg(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            hidden_out_dim: int,
            count_dim: int,
            num_layers: int,
            num_heads:int,
            batch_norm: bool,
            residual: bool,
            readout: str,
        ):
        
        super(GATGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.readout = readout
        self.out_dim = hidden_out_dim

        head_hidden_dim = hidden_dim * num_heads
        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = head_hidden_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
        
        concat_feature_dim = head_hidden_dim + count_dim
        
        self.prepare_gat = nn.Linear(concat_feature_dim, head_hidden_dim)
        
        # GAT message passing layers        
        self.convs = nn.ModuleList([GATLayer(head_hidden_dim, hidden_dim, num_heads, batch_norm=batch_norm, residual=residual) for _ in range(self.num_layers-1)])
        self.convs.append(GATLayer(head_hidden_dim, hidden_out_dim, num_heads=1, batch_norm=batch_norm, residual=residual))
        
        # decoder
        self.decoder = MLPReadout(hidden_out_dim, 1)
        
    def forward(self, x, edge_index, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
            h = self.prepare_gat(h)
        else:
            h = atom_h

        # model step
        for conv in self.convs:
            h = conv(x=h, edge_index=edge_index)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h)
        
        return h