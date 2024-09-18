import gzip
import json
import numpy as np
from os import path as osp
import torch
from torch_geometric import data as tgdata
from tqdm import tqdm
from torch_geometric.datasets import ZINC


def load_zinc_subgraph_dataset(raw_dir, split, hom_files, subset=True, pre_transform=None, transform=None):
    original_data = ZINC(raw_dir, subset=subset, split=split)
    
    all_hom_data = []
    for hom_file in hom_files:
        hom_data = json.load(open(hom_file))
        all_hom_data.append(hom_data)
        
    if split == 'train':
        scale = 0
    elif split == 'val':
        scale = 10000
    elif split == 'test':
        scale = 11000
        
    homcount_dataset = []
    for graph_idx in range(len(original_data)):
        hom_graph_idx = graph_idx + scale
        
        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):
            
            vertex_counts = []
            for hom_list in all_hom_data:
                homcounts = hom_list[str(hom_graph_idx)]['subcounts'][str(v_idx)][:-2]
                vertex_counts += homcounts

            graph_counts.append(vertex_counts)
        
        homcount_dataset.append(
            tgdata.Data(
                x = original_data[graph_idx].x, 
                edge_index = original_data[graph_idx].edge_index, 
                edge_attr = original_data[graph_idx].edge_attr, 
                y = original_data[graph_idx].y,
                homcounts = torch.FloatTensor(graph_counts),
            )
        )
    return homcount_dataset


def load_zinc_homcount_dataset(raw_dir, split, hom_files, idx_list, sub_file, subset=True, pre_transform=None, transform=None):
    original_data = ZINC(raw_dir, subset=subset, split=split)
    
    all_hom_data = []
    for hom_file in hom_files:
        hom_data = json.load(open(hom_file))
        all_hom_data.append(hom_data)
        
    use_subcounts = False
    if len(sub_file) > 0:
        use_subcounts = True
        sub_data = json.load(open(sub_file))
        
    if split == 'train':
        scale = 0
    elif split == 'val':
        scale = 10000
    elif split == 'test':
        scale = 11000
        
    homcount_dataset = []
    for graph_idx in range(len(original_data)):
        hom_graph_idx = graph_idx + scale
        
        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):
            
            vertex_counts = []
            for hom_list in all_hom_data:
                homcounts = hom_list[str(hom_graph_idx)]['homcounts'][str(v_idx)]
                vertex_counts += homcounts
            
            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()
                
            if use_subcounts:
                if "subgraph" in sub_file:
                    sub_counts = sub_data[str(hom_graph_idx)]['subcounts'][str(v_idx)][:-2] # for anchored spasm add subgraph counts
                else:
                    sub_counts = sub_data[str(hom_graph_idx)][str(v_idx)] # for regular spasm add multhom counts
                vertex_counts += sub_counts

            graph_counts.append(vertex_counts)
        
        homcount_dataset.append(
            tgdata.Data(
                x = original_data[graph_idx].x, 
                edge_index = original_data[graph_idx].edge_index, 
                edge_attr = original_data[graph_idx].edge_attr, 
                y = original_data[graph_idx].y,
                homcounts = torch.FloatTensor(graph_counts),
            )
        )
    return homcount_dataset


class ZINCDataset(tgdata.InMemoryDataset):
    def __init__(
        self, root, subset, split, transform=None, pre_transform=None, pre_filter=None, count_type=""
    ):
        self.download_path = f".dataset_src/ZINC_{split}"
        self.count_type = count_type
        self.homcount_idx_list = []
        self.sub_file = ""
                
        if self.count_type == "subgraph":
            self.count_paths = [".dataset_src/zinc_3toC_subgraph.json"]
        elif self.count_type == "homcount":
            self.count_paths = [".dataset_src/zinc_with_homs_c7.json", ".dataset_src/zinc_with_homs_c8.json"]
            self.homcount_idx_list = [0, 3, 11, 15, 31, 46]
        elif self.count_type == "spasm":
            self.count_paths = [".dataset_src/zinc_with_homs_c7.json", ".dataset_src/zinc_with_homs_c8.json"]
            self.homcount_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 22, 24, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
            self.sub_file = ".dataset_src/zinc_3to8C_multhom.json"
        elif self.count_type == "anchoredSpasm":
            self.count_paths = [".dataset_src/zinc_with_anchored_homs_c78_no1wl.json"]
            self.sub_file = ".dataset_src/zinc_3to10C_subgraph.json"
        else:
            raise Exception('Error: Count type not supported')
        
        self.split = split
        
        new_root = osp.join(root, split)
        super(ZINCDataset, self).__init__(
            new_root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list_new = []

        if self.count_type == "subgraph":
            zinc_data = load_zinc_subgraph_dataset(self.download_path, self.split, self.count_paths)
        elif self.count_type in ["homcount", "spasm", "anchoredSpasm"]:
            zinc_data = load_zinc_homcount_dataset(self.download_path, self.split, self.count_paths, self.homcount_idx_list, self.sub_file)
        else:
            raise Exception('Error: Count type not supported')
        
            
        for graph in zinc_data:
            graph_tran = self.pre_transform(graph)
            data_list_new.append(graph_tran)

        data, slices = self.collate(data_list_new)
        torch.save((data, slices), self.processed_paths[0])
