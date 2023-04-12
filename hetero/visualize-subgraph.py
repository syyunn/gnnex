import pickle
from tqdm import tqdm

import numpy as np
import random

from datetime import timedelta

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

edge_index_dicts = {}

for edge_type in edge_types:
    edge_indices = data.edge_index_dict[edge_type]
    edge_index_dict = {(src.item(), dest.item()): i for i, (src, dest) in enumerate(edge_indices.t())}
    edge_index_dicts[edge_type] = edge_index_dict


print("Edge types:", edge_types)
print(len(edge_types))

# # Convert Heterograph to NetworkX MultiGraph
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

# # Create a dictionary mapping edge types to their reverse dictionaries
# edge_type_to_reverse_dict = {
#     'ticker': reverse_tickers,
#     'congressperson': reverse_congresspeople,
#     'committee': reverse_committees,
#     'bill': reverse_bills,
#     'naics': reverse_naics
# }

# # Iterate over edge types and add edges with attributes
# for edge_type, edge_index in data.edge_index_dict.items():
#     edge_attr = data.edge_attr_dict[edge_type]
#     reverse_src_dict = edge_type_to_reverse_dict[edge_type[0]]
#     reverse_dst_dict = edge_type_to_reverse_dict[edge_type[2]]

#     for i, (src, dst) in enumerate(edge_index.t()):
#         # Extract the semantic node labels from corresponding reverse dictionaries
#         src_label = reverse_src_dict.get(src.item())
#         dst_label = reverse_dst_dict.get(dst.item())

#         start_date, end_date = tensor_to_dates(edge_attr[i])
#         G.add_edge(src_label, dst_label, key=edge_type, start_date=start_date, end_date=end_date)

# # Specify the file name where you want to save the pickled graph
# pickle_file = 'networkx_multigraph.pkl'

# # Open the file in binary write mode and pickle the graph
# with open(pickle_file, 'wb') as f:
#     pickle.dump(G, f)

# # Optionally, print a message to indicate that the graph has been pickled successfully
# print(f'MultiGraph G has been pickled and saved to {pickle_file}')


# # Specify the file name where the pickled graph is saved
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


semantic_to_integer_index = {
    'ticker': unique_tickers,
    'congressperson': unique_congresspeople,
    'committee': unique_committees,
    'bill': unique_bills,
    'naics': unique_naics
}


# Add a function to get the edge_mask index from edge_type
def get_edge_mask_index(edge_type, edge_types):
    return edge_types.index(edge_type)

# Update the get_subgraph function
def get_subgraph(G, source_nodes, node_masks, edge_masks, node_mask_threshold, edge_mask_threshold, semantic_to_integer_index, edge_types):
    source_nodes_pass_threshold = set()

    for node in source_nodes: # source node is [congressperson, ticker]
        node_type = G.nodes[node]['node_type']
        
        if node_masks[node_type][semantic_to_integer_index[node_type][node]] > node_mask_threshold:
            source_nodes_pass_threshold.add(node)

    connected_nodes_pass_threshold = set()

    for node in source_nodes_pass_threshold:
        for neighbor, edge_data in G[node].items():
            node_type = G.nodes[node]['node_type']
            neighbor_type = G.nodes[neighbor]['node_type']
            src_idx = semantic_to_integer_index[node_type][node]
            dst_idx = semantic_to_integer_index[neighbor_type][neighbor]

            if node_masks[neighbor_type][semantic_to_integer_index[neighbor_type][neighbor]] > node_mask_threshold:
                for edge_type, edge_attributes in edge_data.items():
                    try:
                        edge_mask_idx = edge_index_dicts[edge_type][(src_idx, dst_idx)]
                        pass
                    except KeyError:
                        edge_mask_idx = edge_index_dicts[edge_type][(dst_idx, src_idx)]
                        pass
                    if edge_masks[edge_type][edge_mask_idx] > edge_mask_threshold:
                        connected_nodes_pass_threshold.add(neighbor)

    subgraph_nodes = source_nodes_pass_threshold | connected_nodes_pass_threshold
    return G.subgraph(subgraph_nodes)


def draw_subgraph(subgraph, node_colors, title=None):
    pos = nx.spring_layout(subgraph, seed=42)
    
    node_legends = []
    for node_type, color in node_colors.items():
        nx.draw(subgraph,
                pos,
                nodelist=[n for n in subgraph.nodes if subgraph.nodes[n]['node_type'] == node_type],
                node_color=color,
                label=node_type)
        node_legends.append(plt.Line2D([], [], color=color, marker='o', linestyle='', label=node_type))
        
    nx.draw_networkx_labels(subgraph, pos, labels={n: n for n in subgraph.nodes})

    plt.title(title)    
    plt.legend(handles=node_legends)
    plt.show()
    pass


node_colors = {
    'ticker': 'red',
    'congressperson': 'blue',
    'committee': 'green',
    'bill': 'orange',
    'naics': 'purple'
}

for congressperson_label, ticker_label in results.keys():
    print(congressperson_label, ticker_label)
    congressperson_id = reverse_congresspeople[congressperson_label]
    ticker_id = reverse_tickers[ticker_label]

    # read the node masks
    node_masks = results[(congressperson_label, ticker_label)]['node_masks']
    # read the edge masks
    edge_masks = results[(congressperson_label, ticker_label)]['edge_masks']

    # Find the sale or purchase edge attributes
    sale_purchase_edges = loaded_G.get_edge_data(congressperson_id, ticker_id)

    if sale_purchase_edges is not None:
        # Add the following inside the loop that iterates over results.keys()
        node_mask_threshold = 0.5
        edge_mask_threshold = 0.5
        node_masks = results[(congressperson_label, ticker_label)]['node_masks']

        source_nodes = [congressperson_id, ticker_id]
        print(source_nodes)
        subgraph = get_subgraph(loaded_G, source_nodes, node_masks, edge_masks, node_mask_threshold, edge_mask_threshold, semantic_to_integer_index, edge_types)
        title = f"Subgraph for {congressperson_label} and {ticker_label} (Filtered)"
        draw_subgraph(subgraph, node_colors, title=title)
        pass


