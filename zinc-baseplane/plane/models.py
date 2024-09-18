from plane import plane
from plane.common_imports import *
from plane.tools import *
from torch_geometric.nn.models import MLP

def get_layer(config: ModelConfig):
    match config.flags_layer:
        case "plane":
            return plane.PlaneLayer(config)
        case "gin":
            return tgnn.GINConv(
                MLP(config.dim, config.dim, factor=2, drop=config.drop_out),
                train_eps=True,
            )
        case "gcn":
            return tgnn.GCNConv(config.dim, config.dim)
        case _:
            raise NotImplementedError


class ModelGraph(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_g = Embedding(config.dim_node_feature, config.dim)
        self.layers = nn.ModuleList(
            [get_layer(config) for _ in range(config.num_layers)]
        )
        self.aggr = tgnn.SumAggregation()
        self.out_final = nn.Sequential(
            nn.LazyLinear(max(1, config.flags_mlp_factor) * config.dim),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.LazyLinear(config.dim_output),
            getattr(nn, config.act_out)(),
        )
        
        self.count_dim = config.count_dim
        self.count_encoder = MLP(in_channels=config.count_dim, hidden_channels=config.dim, out_channels=config.count_dim, num_layers=2)
        concat_feature_dim = config.dim + config.count_dim
        self.in_encoder = MLP(in_channels=concat_feature_dim, hidden_channels=config.dim, out_channels=config.dim, num_layers=2)


    def forward(self, data):
        num_batch = data.num_graphs

        # Initialize node and edge feature
        h_g = self.embed_g(data.x)
        
        # embed homomorphism count features
        if self.count_dim > 0:
            hom_h = data.homcounts
            hom_h = self.count_encoder(hom_h)
            h_g = torch.cat([h_g, hom_h], dim=1)
            h_g = self.in_encoder(h_g)
        
        hist = []
        for i in range(self.config.num_layers):
            match self.config.flags_layer:
                case "plane":
                    h_g = self.layers[i](data, h_g)
                case "gin" | "gcn":
                    h_g = self.layers[i](h_g, data.edge_index)
                case _:
                    raise NotImplementedError

            hist.append(h_g)

        # Aggregate all node features from i-th layer to obtain a graph-level feature
        # Concat then MLP
        # out_h = torch.cat([self.aggr(h_cur, data.batch, dim_size=num_batch) for h_cur in hist], dim=1)
        # hom_h_in = data.homcounts.view(-1, 35)
        # hom_h = self.count_encoder(hom_h_in)
        
        # final_h = torch.cat([out_h, hom_h], dim=1)
        # out = self.out_final(final_h)
        
        out = self.out_final(
            torch.cat(
                [
                    self.aggr(h_cur, data.batch, dim_size=num_batch)
                    for h_cur in hist
                ],
                dim=1,
            )
        )
        return out
