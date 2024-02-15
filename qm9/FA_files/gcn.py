import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.conv import RGCNConv
from models.layers.hsp_gin_layer import instantiate_mlp


# Modes: GC: Graph Classification.
GRAPH_CLASS = "gc"
GRAPH_REG = "gr"


class NetGCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        emb_sizes=None,
        mode=GRAPH_REG,
        drpt_prob=0.5,
        scatter="max",
        device="cpu",
        batch_norm=True,
        layer_norm=False,
        residual_frequency=-1,
        nb_edge_types=1,
    ):
        super(NetGCN, self).__init__()
        if emb_sizes is None:  # Python default handling for mutable input
            emb_sizes = [32, 64, 64]  # The 0th entry is the input feature size.
        self.num_features = num_features
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.drpt_prob = drpt_prob
        self.scatter = scatter
        self.device = device
        self.mode = mode

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual_freq = residual_frequency
        self.nb_edge_types = nb_edge_types

        self.initial_mlp = instantiate_mlp(
            in_channels=num_features,
            out_channels=emb_sizes[0],
            device=device,
            batch_norm=batch_norm,
            final_activation=True,
        )

        self.regression_gate_mlp = instantiate_mlp(
            in_channels=emb_sizes[-1] + num_features + 31, # 5vertex + 6cycle homcounts
            out_channels=1,
            device=device,
            final_activation=False,
            batch_norm=batch_norm,
        )
        self.regression_transform_mlp = instantiate_mlp(
            in_channels=emb_sizes[-1],
            out_channels=1,
            device=device,
            final_activation=False,
            batch_norm=batch_norm,
        )  # No final act
        self.initial_linear = Linear(emb_sizes[0], num_classes).to(device)

        gcn_layers = []
        linears = []
        if self.layer_norm:
            layer_norms = []
        for i in range(self.num_layers):
            in_channel = emb_sizes[i]
            out_channel = emb_sizes[i + 1]
            gcn_layer = RGCNConv(in_channels=in_channel, out_channels=out_channel, num_relations=nb_edge_types, aggr='add').to(device)
            gcn_layers.append(gcn_layer)
            if self.layer_norm:
                layer_norms.append(torch.nn.LayerNorm(emb_sizes[i + 1]).to(device))
            linears.append(Linear(emb_sizes[i + 1], num_classes).to(device))

        self.gcn_modules = ModuleList(gcn_layers)
        self.linear_modules = ModuleList(linears)
        if self.layer_norm:
            self.layer_norms = ModuleList(layer_norms)

    def reset_parameters(self):
        if self.layer_norm:
            for x in self.layer_norms:
                x.reset_parameters()
        if hasattr(self, "initial_mlp"):
            for module in self.initial_mlp:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.gcn_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.linear_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.regression_transform_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.regression_gate_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def pooling(self, x_feat, batch):
        if self.scatter == "max":
            return scatter_max(x_feat, batch, dim=0)[0].to(self.device)
        elif self.scatter == "mean":
            return scatter_mean(x_feat, batch, dim=0).to(self.device)
        else:
            pass

    def forward(self, data):
        x_feat = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        
        fa_edge_index = data.fa_edge_index.to(self.device)
        fa_edge_attr = data.fa_edge_attr.to(self.device)
        
        batch = data.batch.to(self.device)

        x_feat = self.initial_mlp(x_feat)  # Otherwise by an MLP

        if self.residual_freq > 0:
            last_state_list = [x_feat]  # If skip connections are being used

        for idx, value in enumerate(zip(self.gcn_modules, self.linear_modules)):
            gcn_layer, linear_layer = value
            
            if idx == (len(self.gcn_modules) - 1):
                x_feat = gcn_layer(x_feat, fa_edge_index, fa_edge_attr).to(self.device)
            else:
                x_feat = gcn_layer(x_feat, edge_index, edge_attr).to(self.device)
            
            if self.residual_freq > 0:  # Time to introduce a residual
                if self.residual_freq <= idx + 1:
                    x_feat = (
                        x_feat + last_state_list[-self.residual_freq]
                    )  # Residual connection
                last_state_list.append(
                    x_feat
                )  # Add
            
            if self.layer_norm:
                x_feat = torch.relu(
                    self.layer_norms[idx](x_feat)
                )  # Just apply layer norms then. ReLU is crucial.
                # Otherwise Layer Norm freezes
            else:
                x_feat = torch.relu(x_feat)

        h_hom = data.graph_hom.view(-1, 31)

        gate_input = torch.cat([data.x, x_feat, h_hom], dim=-1)
        gate_out = self.regression_gate_mlp(gate_input)

        transform_out = self.regression_transform_mlp(x_feat)
        product = torch.sigmoid(gate_out) * transform_out
        out = scatter_sum(product, batch, dim=0).to(self.device)
        return out

    def log_hop_weights(self, neptune_client, exp_dir):
        # This is a function intended for the SPN, to keep track of weights.
        # For standard GNNs, it is of no use, so we define it as a blank fct
        pass
