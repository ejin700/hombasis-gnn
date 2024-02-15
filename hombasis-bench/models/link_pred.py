import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_heads, count_dim, pe_dim, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GAT, self).__init__()
        self.pe_dim = pe_dim

        head_hidden_dim = hidden_channels * num_heads
        self.encode_h = nn.Linear(in_channels, head_hidden_dim)

        # encode homcounts using positional enconding
        if count_dim > 0:
            self.normalize_hom = PositionalEncoding(pe_dim)
        
        concat_feature_dim = in_channels + count_dim*pe_dim
        self.prepare_gat = nn.Linear(concat_feature_dim, head_hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(head_hidden_dim, hidden_channels, num_heads))
        self.convs.append(GATConv(head_hidden_dim, out_channels, 1))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, use_counts, counts):
        if use_counts:
            count_h = torch.flatten(counts)
            count_h = self.normalize_hom(counts)
            count_size = counts.size()
            count_h = count_h.view((count_size[0], count_size[1]*self.pe_dim))

            x = torch.cat([x, count_h], dim=1)
            x = self.prepare_gat(x)
            
        else:
            x = self.encode_h(x)
        
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    

class GCN(torch.nn.Module):
    def __init__(self, in_channels, count_dim, pe_dim, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.pe_dim = pe_dim

        # encode homcounts using positional encoding
        if count_dim > 0:
            self.normalize_hom = PositionalEncoding(pe_dim)
        
        concat_feature_dim = in_channels + count_dim*pe_dim

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(concat_feature_dim, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, use_counts, counts):
        if use_counts:
            count_h = torch.flatten(counts)
            count_h = self.normalize_hom(counts)
            count_size = counts.size()
            count_h = count_h.view((count_size[0], count_size[1]*self.pe_dim))
            x = torch.cat([x, count_h], dim=1)
        
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, count_dim, pe_dim, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()
        self.pe_dim = pe_dim

        # encode homcounts using positional encoding
        if count_dim > 0:
            self.normalize_hom = PositionalEncoding(pe_dim)
        
        concat_feature_dim = in_channels + count_dim*pe_dim

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(concat_feature_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, use_counts, counts):
        if use_counts:
            count_h = torch.flatten(counts)
            count_h = self.normalize_hom(counts)
            count_size = counts.size()
            count_h = count_h.view((count_size[0], count_size[1]*self.pe_dim))
            x = torch.cat([x, count_h], dim=1)

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x