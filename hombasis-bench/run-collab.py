import argparse
import random
import os
import yaml
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import torch_geometric.transforms as T
import torch_geometric.seed
from ogb.linkproppred import Evaluator

from data.get_data import load_collab_counts, load_collab_multsum_counts
from models.link_pred import GAT, GCN, SAGE


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t, data.use_counts, data.counts)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t, data.use_counts, data.counts)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(data.x, data.full_adj_t, data.use_counts, data.counts)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--num_heads", type=int, default=8)
    args = parser.parse_args()
    print(args)

    default_seed = args.seed
    torch.manual_seed(default_seed)
    torch.cuda.manual_seed(default_seed)
    torch.cuda.manual_seed_all(default_seed)
    np.random.seed(default_seed)
    random.seed(default_seed)
    torch_geometric.seed.seed_everything(default_seed)
    
    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)
    
    project = conf['project']
    group = conf['group']
    name = conf['name']
    description = conf['description']
    use_counts = conf['use_counts']
    count_files = conf['count_files']
    idx_list = conf['idx_list']
    model_name = conf['model_name']
    pe_dim = conf['pe_dim']
    
    wandb.init(
        project=project,
        group=group,
        name=name
    )
    
    wandb.config.update(args)
    print(default_seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data', 'collab-data')
    
    # load data with counts
    if 'path' in args.config:
        count_type = args.config[-10:-5]
        print(count_type)
        dataset, data = load_collab_multsum_counts(data_dir, use_counts, count_type, count_files, idx_list)
        
    else:
        dataset, data = load_collab_counts(data_dir, use_counts, count_files, idx_list)
        
    print(data.count_dim)

    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)

    split_edge = dataset.get_edge_split()

    data.full_adj_t = data.adj_t

    data = data.to(device)

    if model_name == 'GCN':
        model = GCN(data.num_features, data.count_dim, pe_dim, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif model_name == 'GAT':
        model = GAT(data.num_features, args.num_heads, data.count_dim, pe_dim, 32,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif model_name == "SAGE":
        model = SAGE(data.num_features, data.count_dim, pe_dim, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    else:
        print('model not supported')

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                            args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)

    valid_curve = []
    best_val_epoch = 0
    test_curve = []
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, predictor, data, split_edge, optimizer,
                        args.batch_size)

        results = test(model, predictor, data, split_edge, evaluator,
                            args.batch_size)

        valid_curve.append(results['Hits@50'][1])
        test_curve.append(results['Hits@50'][2])
        
        wandb.log({
            "loss": loss,
            "train_hits_50": results['Hits@50'][0],
            "valid_hits_50": results['Hits@50'][1],
            "test_hits_50": results['Hits@50'][2],
        })
        
        
    best_val_epoch = np.argmax(np.array(valid_curve))
    print('best epoch')
    print(best_val_epoch)
    print('best test')
    print(test_curve[best_val_epoch])
    
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    
   # save results
    wandb.log({
        'best': best_val_epoch,
        'best_val': valid_curve[best_val_epoch],
        'best_test': test_curve[best_val_epoch],
        'description': description,
        'final_epoch': epoch,
        'total_params': total_params,
        'pe_dim': args.pe_dim,
        
    })
    
    
    wandb.finish()

