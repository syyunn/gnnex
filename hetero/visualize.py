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

# Convert Heterograph to NetworkX MultiGraph
# G = nx.MultiGraph()

# # Add ticker nodes
# ticker_nodes = [(k, {'node_type': 'ticker', 'label': v}) for k, v in unique_tickers.items()]
# G.add_nodes_from(ticker_nodes)

# # Add congressperson nodes
# congressperson_nodes = [(k, {'node_type': 'congressperson', 'label': v}) for k, v in unique_congresspeople.items()]
# G.add_nodes_from(congressperson_nodes)

# # Add committee nodes
# committee_nodes = [(k, {'node_type': 'committee', 'label': v}) for k, v in unique_committees.items()]
# G.add_nodes_from(committee_nodes)

# # Add bill nodes
# bill_nodes = [(k, {'node_type': 'bill', 'label': v}) for k, v in unique_bills.items()]
# G.add_nodes_from(bill_nodes)

# # Add naics nodes
# naics_nodes = [(k, {'node_type': 'naics', 'label': v}) for k, v in unique_naics.items()]
# G.add_nodes_from(naics_nodes)

# # definse some utils to get attributes date
# from datetime import date, timedelta

# def days_to_date(days_elapsed, reference_date=date(2016, 1, 1)):
#     return reference_date + timedelta(days=int(days_elapsed))

# def tensor_to_dates(edge_attr_tensor):
#     start_date = days_to_date(edge_attr_tensor[0])
#     end_date = days_to_date(edge_attr_tensor[1])
#     return start_date, end_date

# # Iterate over edge types and add edges with attributes
# for edge_type, edge_index in data.edge_index_dict.items():
#     edge_attr = data.edge_attr_dict[edge_type]

#     for i, (src, dst) in enumerate(edge_index.t()):
#         # Extract the semantic node labels from reverse dictionaries
#         src_label = reverse_tickers.get(src.item()) or reverse_congresspeople.get(src.item()) or reverse_committees.get(src.item()) or reverse_bills.get(src.item()) or reverse_naics.get(src.item())
#         dst_label = reverse_tickers.get(dst.item()) or reverse_congresspeople.get(dst.item()) or reverse_committees.get(dst.item()) or reverse_bills.get(dst.item()) or reverse_naics.get(dst.item())

#         start_date, end_date = tensor_to_dates(edge_attr[i])
#         G.add_edge(src_label, dst_label, key=edge_type, start_date=start_date, end_date=end_date)

# # Specify the file name where you want to save the pickled graph
# pickle_file = 'networkx_multigraph.pkl'

# # Open the file in binary write mode and pickle the graph
# with open(pickle_file, 'wb') as f:
#     pickle.dump(G, f)

# # Optionally, print a message to indicate that the graph has been pickled successfully
# print(f'MultiGraph G has been pickled and saved to {pickle_file}')


# Specify the file name where the pickled graph is saved
pickle_file = 'networkx_multigraph.pkl'

# Open the file in binary read mode and unpickle the graph
with open(pickle_file, 'rb') as f:
    loaded_G = pickle.load(f)

# Optionally, print a message to indicate that the graph has been unpickled successfully
print('MultiGraph G has been unpickled from', pickle_file)

# Now you can use the loaded_G as needed

import pickle

with open("node_edge_masks_results.pkl", "rb") as f:
    results = pickle.load(f)


for congressperson_label, ticker_lable in results.keys():
    print(congressperson_label, ticker_lable)
    congressperson_label = reverse_congresspeople[congressperson_label]
    ticker_label = reverse_tickers[ticker_lable]
    pass
pass


