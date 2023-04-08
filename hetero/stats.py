import pickle
from tqdm import tqdm


# Open the file in binary read mode and unpickle the data
with open('/home/gridsan/syun/gnnex/data/hetero_graph_data.pkl', "rb") as f:
    loaded_data = pickle.load(f)

# Extract the data from the loaded dictionary
data = loaded_data["hetero_graph"]
unique_tickers = loaded_data["unique_tickers"]
unique_congresspeople = loaded_data["unique_congresspeople"]
unique_committees = loaded_data["unique_committees"]
unique_bills = loaded_data["unique_bills"]
unique_naics = loaded_data["unique_naics"]

import torch

# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))

from torch import Tensor
from torch_geometric.data import HeteroData
from typing import List

# Assign consecutive indices to each node type
data['congressperson'].node_id = torch.arange(len(unique_congresspeople))
data['committee'].node_id = torch.arange(len(unique_committees))
data['ticker'].node_id = torch.arange(len(unique_tickers))
data['bill'].node_id = torch.arange(len(unique_bills))
data['naics'].node_id = torch.arange(len(unique_naics))

# Print the updated data
print("Node IDs have been assigned to each node type.")
print(data)

def compute_average_degree(hetero_data):
    avg_degrees = {}
    for node_type in hetero_data.node_types:
        total_degree = 0
        num_nodes = hetero_data[node_type].num_nodes
        for rel in hetero_data.edge_types:
            if rel[0] == node_type:
                total_degree += hetero_data[rel].edge_index.size(1)
            if rel[2] == node_type:
                total_degree += hetero_data[rel].edge_index.size(1)
        avg_degree = total_degree / num_nodes
        avg_degrees[node_type] = avg_degree
    return avg_degrees

# Compute the average degree for each node type in the heterogeneous graph
avg_degrees = compute_average_degree(data)

# Print the results
for node_type, avg_degree in avg_degrees.items():
    print(f"Average degree of node type '{node_type}': {avg_degree:.2f}")
