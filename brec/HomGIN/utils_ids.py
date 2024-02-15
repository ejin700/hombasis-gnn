import torch
import numpy as np
from torch_geometric.utils import remove_self_loops


# def subgraph_counts2ids(count_fn, data, subgraph_dicts, subgraph_params):
    
#     #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####
    
#     if hasattr(data, 'edge_features'):
#         edge_index, edge_features = remove_self_loops(data.edge_index, data.edge_features)
#         setattr(data, 'edge_features', edge_features)
#     else:
#         edge_index = remove_self_loops(data.edge_index)[0]
             
#     num_nodes = data.x.shape[0]
#     identifiers = None
#     for subgraph_dict in subgraph_dicts:
#         kwargs = {'subgraph_dict': subgraph_dict, 
#                   'induced': subgraph_params['induced'],
#                   'num_nodes': num_nodes,
#                   'directed': subgraph_params['directed']}
#         counts = count_fn(edge_index, **kwargs)
#         identifiers = counts if identifiers is None else torch.cat((identifiers, counts),1) 
#     setattr(data, 'edge_index', edge_index)
#     setattr(data, 'identifiers', identifiers.long())
    
#     return data


def homcount_counts2ids(data, homcount_dict):
    #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####
    
    if hasattr(data, 'edge_features'):
        edge_index, edge_features = remove_self_loops(data.edge_index, data.edge_features)
        setattr(data, 'edge_features', edge_features)
    else:
        edge_index = remove_self_loops(data.edge_index)[0]
        
    # parse homcount dict
    num_nodes = data.x.shape[0]
    homcounts = []
    for i in range(num_nodes):
        try:
            v_count = homcount_dict[str(i)]
        except: # there are disconnected nodes
            v_count = [0]*30 # 30 is dim of 5v homcount
           
        homcounts.append(v_count)
    
    identifiers = torch.tensor(homcounts)

    setattr(data, 'edge_index', edge_index)
    setattr(data, 'identifiers', identifiers.long())
    
    return data
