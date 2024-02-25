import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx, to_networkx
import os
from tqdm import tqdm
import json

torch_geometric.seed_everything(2022)
DATA_SPLIT_NAME = "brec_cfi"

def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="hom_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return [f"{DATA_SPLIT_NAME}.npy"]

    @property
    def processed_file_names(self):
        return [f"{DATA_SPLIT_NAME}.pt"]
    
    def normalize_graph(curr_graph):
        split = np.split(curr_graph, [1], axis=2)

        adj = np.squeeze(split[0], axis=2)
        deg = np.sqrt(np.sum(adj, 0))
        deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
        normal = np.diag(deg)
        norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
        ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
        spred_adj = np.multiply(ones, norm_adj)
        labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
        
        return np.add(spred_adj, labels)

    def process(self):

        homcount_file = os.path.join(self.root, f'raw/{DATA_SPLIT_NAME}_v5_counts.json')
        with open(homcount_file, 'r') as f:
            homcount_data = json.load(f)

        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data = []
        g_idx = 0
        for g in data_list:
            g_networkx = nx.from_graph6_bytes(g)
            edge = nx.to_numpy_array(g_networkx)
            node = np.eye(g_networkx.number_of_nodes())
            
            num_nodes = g_networkx.number_of_nodes()
            homcounts = []
            for v_idx in range(num_nodes):
                try:
                    v_count = homcount_data[str(g_idx)]['homcounts'][str(v_idx)]
                except:
                    v_count = [0] * 30
                    
                homcounts.append(v_count)
            
            
            homcount_np = np.array(homcounts)            
            graph = np.empty((num_nodes, num_nodes, 31))
            
            for i in range(1, 31):
                vHom_counts = homcount_np[:, i-1]
                
                graph[:, :, i] =+ np.diag(vHom_counts)

                if max(vHom_counts) > 0:
                    graph[:,:,i] /= max(vHom_counts)


            graph[:, :, 0] = (edge + node)
            graph = BRECDataset.normalize_graph(graph)

            gt = np.transpose(graph, [2,0,1])
            graph = torch.tensor(gt, dtype=torch.float32)

            data.append(graph)
            
            g_idx += 1
                        
        self.data = data

        torch.save(self.data, self.processed_paths[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


def main():
    dataset = BRECDataset()
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
