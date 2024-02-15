import os
import json
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC

from ogb.linkproppred import PygLinkPropPredDataset


def load_zinc_subcount_dataset(name, sub_file, idx_list, root):    
    if name == "ZINC":
        train_data, val_data, test_data = load_zinc_dataset(name, root)
        
    original_data = train_data + val_data + test_data

    sub_data = json.load(open(os.path.join(root, sub_file)))
        
    subcount_dataset = []
    for graph_idx in range(len(original_data)):

        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):

            vertex_counts = sub_data[str(graph_idx)]['subcounts'][str(v_idx)]
            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()
                
            graph_counts.append(vertex_counts)
            
        subcount_dataset.append(
            Data(
                x = original_data[graph_idx].x, 
                edge_index = original_data[graph_idx].edge_index, 
                edge_attr = original_data[graph_idx].edge_attr, 
                y = original_data[graph_idx].y,
                counts = torch.Tensor(graph_counts),
            )
        )

    assert len(subcount_dataset) == len(original_data)

    new_train_data = subcount_dataset[:len(train_data)]
    val_split = len(train_data) + len(val_data)
    new_val_data = subcount_dataset[len(train_data):val_split]
    new_test_data = subcount_dataset[val_split:]
    
    subcount_dim = subcount_dataset[0].counts.size()[1]
    
    return new_train_data, new_val_data, new_test_data, subcount_dim


def load_zinc_homcount_dataset(name, hom_files, idx_list, root):    
    if name == "ZINC":
        train_data, val_data, test_data = load_zinc_dataset(name, root)
        
    original_data = train_data + val_data + test_data
    
    all_hom_data = []
    for hom_file in hom_files:
        hom_path = os.path.join(root, hom_file)
        hom_data = json.load(open(hom_path))
        all_hom_data.append(hom_data)
        
    homcount_dataset = []
    for graph_idx in range(len(original_data)):
        
        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):
            
            vertex_counts = []
            for hom_list in all_hom_data:
                homcounts = hom_list[str(graph_idx)]['homcounts'][str(v_idx)]
                vertex_counts += homcounts
            
            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()

            graph_counts.append(vertex_counts)
                        
        homcount_dataset.append(
            Data(
                x = original_data[graph_idx].x, 
                edge_index = original_data[graph_idx].edge_index, 
                edge_attr = original_data[graph_idx].edge_attr, 
                y = original_data[graph_idx].y,
                counts = torch.Tensor(graph_counts),
            )
        )

    assert len(homcount_dataset) == len(original_data)

    new_train_data = homcount_dataset[:len(train_data)]
    val_split = len(train_data) + len(val_data)
    new_val_data = homcount_dataset[len(train_data):val_split]
    new_test_data = homcount_dataset[val_split:]
    
    homcount_dim = homcount_dataset[0].counts.size()[1]
    
    return new_train_data, new_val_data, new_test_data, homcount_dim


def load_zinc_subhom_dataset(name, hom_files, idx_list, sub_file, root):    
    if name == "ZINC":
        train_data, val_data, test_data = load_zinc_dataset(name, root)
        
    original_data = train_data + val_data + test_data
    
    all_hom_data = []
    for hom_file in hom_files:
        hom_path = os.path.join(root, hom_file)
        hom_data = json.load(open(hom_path))
        all_hom_data.append(hom_data)
        
    sub_data = json.load(open(os.path.join(root, sub_file)))
    homcount_dataset = []
    for graph_idx in range(len(original_data)):
        
        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):
            
            vertex_counts = []
            for hom_list in all_hom_data:
                homcounts = hom_list[str(graph_idx)]['homcounts'][str(v_idx)]
                vertex_counts += homcounts
                
            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()

            if "anchor" in sub_file:
                sub_counts = sub_data[str(graph_idx)]['subcounts'][str(v_idx)][:-2] # for anchored spasm
            else:
                sub_counts = sub_data[str(graph_idx)][str(v_idx)] # for spasm
            vertex_counts += sub_counts
            graph_counts.append(vertex_counts)
                        
        homcount_dataset.append(
            Data(
                x = original_data[graph_idx].x, 
                edge_index = original_data[graph_idx].edge_index, 
                edge_attr = original_data[graph_idx].edge_attr, 
                y = original_data[graph_idx].y,
                counts = torch.Tensor(graph_counts),
            )
        )

    assert len(homcount_dataset) == len(original_data)

    new_train_data = homcount_dataset[:len(train_data)]
    val_split = len(train_data) + len(val_data)
    new_val_data = homcount_dataset[len(train_data):val_split]
    new_test_data = homcount_dataset[val_split:]
    
    homcount_dim = homcount_dataset[0].counts.size()[1]
    
    return new_train_data, new_val_data, new_test_data, homcount_dim

        
def load_zinc_dataset(name, root, subset=True, pre_transform=None, transform=None):
    # directory where raw data will be downloaded
    raw_dir = os.path.join(root, name)
    print(raw_dir)

    train_data = ZINC(raw_dir, subset=subset, split='train', pre_transform=pre_transform, transform=transform)
    val_data = ZINC(raw_dir, subset=subset, split='val', pre_transform=pre_transform, transform=transform)
    test_data = ZINC(raw_dir, subset=subset, split='test', pre_transform=pre_transform, transform=transform)

    if subset:
        assert len(train_data) == 10000
        assert len(val_data) == 1000
        assert len(test_data) == 1000
    else:
        assert len(train_data) == 220011
        assert len(val_data) == 24445
        assert len(test_data) == 5000

    return train_data, val_data, test_data


def load_collab_counts(root, use_counts, hom_files=None, idx_list=None):
    dataset = PygLinkPropPredDataset(name = "ogbl-collab", root=root)
    data = dataset[0]
    data.use_counts = use_counts
    data.count_dim = 0

    all_hom_data = []
    for hom_file in hom_files:
        hom_path = os.path.join(root, hom_file)
        hom_data = json.load(open(hom_path))
        all_hom_data.append(hom_data)
    
    graph_counts = []
    for v_idx in range(dataset.num_nodes):
        
        vertex_counts = []
        for hom_list in all_hom_data:
            homcounts = hom_list[str(v_idx)]
            vertex_counts += homcounts
                    
        if len(idx_list) > 0:
            vertex_counts = np.array(vertex_counts)[idx_list].tolist()

        graph_counts.append(vertex_counts)
    
    counts = torch.Tensor(graph_counts)
    data.counts = counts
    
    if use_counts:
        data.count_dim = len(vertex_counts)
            
    return dataset, data
