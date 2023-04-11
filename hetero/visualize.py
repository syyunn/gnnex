import pickle
from tqdm import tqdm

import numpy as np
import random

import torch

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Open the file in binary read mode and unpickle the data
# with open('/home/gridsan/syun/gnnex/data/hetero_graph_data.pkl', "rb") as f:
with open('./data/hetero_graph_data.pkl', "rb") as f:
    loaded_data = pickle.load(f)

# Extract the data from the loaded dictionary
data = loaded_data["hetero_graph"]
# Uuniqe_x are dictionary that maps semantic to integer index of them - like 'hconres1-115' to 0
unique_tickers = loaded_data["unique_tickers"] 
unique_congresspeople = loaded_data["unique_congresspeople"]
unique_committees = loaded_data["unique_committees"]
unique_bills = loaded_data["unique_bills"]
unique_naics = loaded_data["unique_naics"]

# Assign consecutive indices to each node type
data['congressperson'].node_id = torch.arange(len(unique_congresspeople))
data['committee'].node_id = torch.arange(len(unique_committees))
data['ticker'].node_id = torch.arange(len(unique_tickers))
data['bill'].node_id = torch.arange(len(unique_bills))
data['naics'].node_id = torch.arange(len(unique_naics))

# Print the updated data
print("Node IDs have been assigned to each node type.")
print(data)
print(data.node_types)

# Create reverse mappings for all node types
reverse_tickers = {v: k for k, v in unique_tickers.items()}
reverse_congresspeople = {v: k for k, v in unique_congresspeople.items()}
reverse_committees = {v: k for k, v in unique_committees.items()}
reverse_bills = {v: k for k, v in unique_bills.items()}
reverse_naics = {v: k for k, v in unique_naics.items()}

# Collect edge_types 
edge_types = []
# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    edge_types.append(edge_type)

print("Edge types:", edge_types)
print(len(edge_types))

nx_graph = to_networkx(data)

pass