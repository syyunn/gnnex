import pickle
from tqdm import tqdm

import numpy as np
import random

from datetime import timedelta

import torch

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

from datetime import datetime as dt

def is_date_within_range(start_date, end_date, date_lower_bound, date_upper_bound):
    start_date_days = days_since_2016_01_01(start_date)
    end_date_days = days_since_2016_01_01(end_date)

    return (start_date_days <= date_upper_bound and end_date_days >= date_lower_bound)

def days_since_2016_01_01(date_obj):
    reference_date = dt(2016, 1, 1)
    date_obj_datetime = dt(date_obj.year, date_obj.month, date_obj.day)
    delta = date_obj_datetime - reference_date
    return delta.days

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

semantic_to_integer_index = {
    'ticker': unique_tickers,
    'congressperson': unique_congresspeople,
    'committee': unique_committees,
    'bill': unique_bills,
    'naics': unique_naics
}

integer_to_semantic_index = {
    'ticker': reverse_tickers,
    'congressperson': reverse_congresspeople,
    'committee': reverse_committees,
    'bill': reverse_bills,
    'naics': reverse_naics
}

# Update the get_subgraph function
def get_subgraph(G, source_nodes, node_masks, edge_masks, node_mask_threshold, edge_mask_threshold, semantic_to_integer_index, edge_types):
    source_nodes_pass_threshold = set()

    for node in source_nodes: # source node is [congressperson, ticker]
        node_type = G.nodes[node]['node_type']
        
        if node_masks[node_type][semantic_to_integer_index[node_type][node]] >= node_mask_threshold:
            source_nodes_pass_threshold.add(node)

    connected_nodes_pass_threshold = set()

    for node in source_nodes_pass_threshold:
        for neighbor, edge_data in G[node].items():
            node_type = G.nodes[node]['node_type']
            neighbor_type = G.nodes[neighbor]['node_type']
            src_idx = semantic_to_integer_index[node_type][node]
            dst_idx = semantic_to_integer_index[neighbor_type][neighbor]

            if node_masks[neighbor_type][semantic_to_integer_index[neighbor_type][neighbor]] >= node_mask_threshold:
                for edge_type, edge_attributes in edge_data.items():
                    try:
                        edge_mask_idx = edge_index_dicts[edge_type][(src_idx, dst_idx)]
                        pass
                    except KeyError:
                        edge_mask_idx = edge_index_dicts[edge_type][(dst_idx, src_idx)]
                        pass
                    if edge_masks[edge_type][edge_mask_idx] >= edge_mask_threshold:
                        connected_nodes_pass_threshold.add(neighbor)

    subgraph_nodes = source_nodes_pass_threshold | connected_nodes_pass_threshold
    return G.subgraph(subgraph_nodes)


def draw_subgraph(subgraph, node_colors, shapes, title=None, congressperson_label=None, ticker_label=None):
    

    pos = nx.spring_layout(subgraph, seed=42, k=0.8)
    
    node_legends = []
    for node_type, (color, shape) in zip(node_colors.keys(), zip(node_colors.values(), shapes.values())):
        nx.draw_networkx_nodes(subgraph,
                               pos,
                               nodelist=[n for n in subgraph.nodes if subgraph.nodes[n].get('node_type') == node_type],
                               node_color=color,
                               node_shape=shape,
                               label=node_type)
        node_legends.append(plt.Line2D([], [], color=color, marker=shape, linestyle='', label=node_type))
    
    nx.draw_networkx_edges(subgraph, pos)
    nx.draw_networkx_labels(subgraph, pos, labels={n: subgraph.nodes[n].get('name', n) for n in subgraph.nodes})
    
    # Annotate the congressperson and ticker nodes
    if congressperson_label is not None and ticker_label is not None:
        for node, coords in pos.items():
            if node == congressperson_label:
                plt.annotate("Target Congressperson", (coords[0], coords[1]), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=9, color='blue')
            if node == ticker_label:
                plt.annotate("Target Ticker", (coords[0], coords[1]), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=9, color='red')

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

def get_thresholded_subgraph(G, node_masks, edge_masks, node_mask_threshold, edge_mask_threshold, semantic_to_integer_index, edge_types, edge_index_dicts, target_nodes, reference_date):
    # Create a new MultiGraph to store the filtered subgraph
    subgraph = nx.MultiGraph()

    # Add the edge between the target congressperson and target ticker
    subgraph.add_edge(target_nodes[0], target_nodes[1], key=('congressperson', 'buy-sell', 'ticker'), **G[target_nodes[0]][target_nodes[1]][('congressperson', 'buy-sell', 'ticker')])

    # Prepare a set to store the nodes and edges that pass the threshold
    thresholded_nodes = set()
    thresholded_edges = set()

    # Add target nodes with their attributes
    for target_node in target_nodes:
        subgraph.add_node(target_node, **G.nodes[target_node])

    # Find the ticker's 1-hop 'naics' neighbor
    ticker = target_nodes[1]
    naics_neighbor = None

    for neighbor, _ in G[ticker].items():
        if G.nodes[neighbor]['node_type'] == 'naics':
            naics_neighbor = neighbor
            break
    
    if naics_neighbor is not None:
        # Add the naics neighbor node of target ticker with its attributes
        subgraph.add_node(naics_neighbor, **G.nodes[naics_neighbor])

        # Add the edge between target ticker and naics neighbor
        edge_data = G[ticker][naics_neighbor][('ticker', 'classified', 'naics')]
        subgraph.add_edge(ticker, naics_neighbor, key=('ticker', 'classified', 'naics'), **edge_data)

    # target_ticker-target_naics  congresperson 
    # add top_k edges for for target_naics for the edge type ('naics', 'classified', 'ticker')
    edge_types = [('ticker', 'classified', 'naics'), ('naics', 'rev_classified', 'ticker')]
    top_k = 20 #At most 10 edges in the same industry
    for edge_type in edge_types:
        edge_mask = edge_masks[edge_type]

        import heapq
        # Find the indices related to the target naics_neighbor
        target_naics_indices = [idx for (src, dst), idx in edge_index_dicts[edge_type].items() if (edge_type[0] == 'ticker' and integer_to_semantic_index[edge_type[2]][dst] == naics_neighbor) or (edge_type[2] == 'ticker' and integer_to_semantic_index[edge_type[0]][src] == naics_neighbor)]

        # Get the edge_mask values for the target_naics_indices
        target_naics_edge_mask_values = edge_mask[target_naics_indices]
        
        # Get the top 10 indices for the target naics_neighbor
        top_10_indices = heapq.nlargest(top_k, range(len(target_naics_edge_mask_values)), target_naics_edge_mask_values.take)
        top_10_target_naics_indices = np.array(target_naics_indices)[top_10_indices]
        
        src_dst_pairs = [edge for edge, idx in edge_index_dicts[edge_type].items() if idx in top_10_target_naics_indices]
        thresholded_edges.update([(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs])

    # Initialize a variable to store the specified edge types
    lobbies_edge_types = [('ticker', 'lobbies_on', 'bill'), ('bill', 'rev_lobbies_on', 'ticker')]

    # Get the ticker nodes from the thresholded_edges set
    ticker_nodes = {src for src, dst, edge_type in thresholded_edges if edge_type[0] == 'ticker'}
    ticker_nodes.update({dst for src, dst, edge_type in thresholded_edges if edge_type[2] == 'ticker'})
    print("Ticker nodes: ", ticker_nodes)

    # Initialize a variable for the top_l edges and look_back period
    top_l = 1 # lobbying edge
    look_back = 365

    for edge_type in lobbies_edge_types:
        edge_mask = edge_masks[edge_type]
        
        # Find the indices related to the ticker nodes in the subgraph
        ticker_indices = [idx for (src, dst), idx in edge_index_dicts[edge_type].items() if (edge_type[0] == 'ticker' and integer_to_semantic_index['ticker'][src] in ticker_nodes) or (edge_type[2] == 'ticker' and integer_to_semantic_index['ticker'][dst] in ticker_nodes)]

        # Get the edge_mask values and edge_attr values for the ticker_indices
        ticker_edge_mask_values = edge_mask[ticker_indices]

        valid_date_indices = [idx for idx, (src, dst) in enumerate([(src, dst) for (src, dst), idx in edge_index_dicts[edge_type].items() if idx in ticker_indices]) if (days_since_2016_01_01(G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['start_date']) >= (reference_date - look_back)) & (days_since_2016_01_01(G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['start_date']) <= reference_date)]

        # Get the top_l indices for the valid date window
        top_l_indices = heapq.nlargest(top_l, valid_date_indices, ticker_edge_mask_values.take)
        top_l_ticker_indices = np.array(ticker_indices)[top_l_indices]

        src_dst_pairs = [edge for edge, idx in edge_index_dicts[edge_type].items() if idx in top_l_ticker_indices]
        thresholded_edges.update([(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs])

        print(f"Added edges for edge type {edge_type}: {[(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs]}")

    # Find edges connecting bills and committees within the edge type ('bill', 'assignment', 'committee')
    edge_type = ('bill', 'assigned_to', 'committee')
    edge_mask = edge_masks[edge_type]

    # Get the bill nodes from the thresholded_edges set
    bill_nodes = {src for src, dst, edge_type in thresholded_edges if edge_type[0] == 'bill'}
    bill_nodes.update({dst for src, dst, edge_type in thresholded_edges if edge_type[2] == 'bill'})
    print("Bill nodes: ", bill_nodes)

    # Find the indices related to the bill nodes in the subgraph
    bill_indices = [idx for (src, dst), idx in edge_index_dicts[edge_type].items() if integer_to_semantic_index['bill'][src] in bill_nodes]

    # Find the edges that have overlap with the 1-year window
    # valid_date_indices = [idx for idx, (src, dst) in enumerate([(src, dst) for (src, dst), idx in edge_index_dicts[edge_type].items() if idx in bill_indices]) if is_date_within_range(G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['start_date'], G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['end_date'], reference_date - look_back, reference_date)]
    # valid_date_indices = [idx for (src, dst), idx in edge_index_dicts[edge_type].items() if idx in bill_indices]
    valid_date_indices = bill_indices # it doesnt need to filter since bills are assinged to committee is specifiable.

    src_dst_pairs = [edge for edge, idx in edge_index_dicts[edge_type].items() if idx in valid_date_indices]
    thresholded_edges.update([(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs])    

    print(f"Added edges for edge type {edge_type}: {[(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs]}")

    # Find edges connecting the target congressperson and committees within the edge type ('congressperson', 'assignment', 'committee')
    edge_type = ('congressperson', 'assignment', 'committee')
    edge_mask = edge_masks[edge_type]

    # Get the target congressperson node
    target_congressperson = target_nodes[0]

    # Get the committee nodes from the thresholded_edges set
    committee_nodes = {dst for src, dst, edge_type in thresholded_edges if edge_type[2] == 'committee'}

    # Find the indices related to the target congressperson and committee nodes in the subgraph
    congressperson_indices = [idx for (src, dst), idx in edge_index_dicts[edge_type].items() if (integer_to_semantic_index['congressperson'][src] == target_congressperson) and (integer_to_semantic_index['committee'][dst] in committee_nodes)]

    # Find the edges that have overlap with the 1-year window
    # valid_date_indices = [idx for idx, (src, dst) in enumerate([(src, dst) for (src, dst), idx in edge_index_dicts[edge_type].items() if idx in congressperson_indices]) if is_date_within_range(G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['start_date'], G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['end_date'], reference_date - look_back, reference_date)]
    # valid_date_indices = [idx for idx in congressperson_indices if is_date_within_range(G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['start_date'], G[integer_to_semantic_index[edge_type[0]][src]][integer_to_semantic_index[edge_type[2]][dst]][edge_type]['end_date'], reference_date - look_back, reference_date)
    valid_date_indices = congressperson_indices

    src_dst_pairs = [edge for edge, idx in edge_index_dicts[edge_type].items() if idx in valid_date_indices]
    thresholded_edges.update([(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs])    

    print(f"Added edges for edge type {edge_type}: {[(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs]}")

    # Add filtered edges
    for src, dst, edge_type in thresholded_edges:   
        # if len(src) < 30 and len(dst) < 30:

        edge_data = G[src][dst][edge_type]
        subgraph.add_edge(src, dst, key=edge_type, **edge_data)

        # Add the src and dst nodes with their attributes
        src_node_data = G.nodes[src]
        dst_node_data = G.nodes[dst]
        subgraph.add_node(src, **src_node_data)
        subgraph.add_node(dst, **dst_node_data)

    return subgraph
    # for node_type, node_mask in node_masks.items():
    #     node_indices = np.where(node_mask >= node_mask_threshold[node_type])[0]
    #     thresholded_nodes.update([integer_to_semantic_index[node_type][i] for i in node_indices])
    
    # # # Add target nodes to the set
    # # for node_label in target_nodes:
    # #     thresholded_nodes.add(node_label)

    # for edge_type, edge_mask in edge_masks.items():
    #     edge_attr = data[edge_type].edge_attr

    #     if edge_type == ('ticker', 'classified', 'naics') or edge_type == ('naics', 'rev_classified', 'ticker'):
    #         # # edge_indices_mask = np.where(edge_mask >= edge_mask_threshold[edge_type])[0]
    #         # import heapq
    #         # # Get the indices of the top 10 largest edge_mask values
    #         # top_10_indices = heapq.nlargest(top_k, range(len(edge_mask)), edge_mask.take)
    #         # edge_indices_mask = np.array(top_10_indices)

    #         # edge_indices = edge_indices_mask
    #         pass
    #     else:
    #         if edge_type == ('ticker', 'lobbies_on', 'bill') or ('bill', 'rev_lobbies_on', 'ticker'):
    #             top_k = 1000
    #         else:
    #             top_k = 100

    #         import heapq

    #         look_back = 365
    #         # edge_indices_date = np.where(edge_attr[:, 0] < reference_date)[0] # start date should be "earlier" than reference date
    #         edge_indices_date = np.where((edge_attr[:, 0] >= (reference_date - look_back)) & (edge_attr[:, 0] <= reference_date))[0]

    #         # edge_indices_date = np.where(edge_attr[:, 0] <= reference_date)[0]

    #         # edge_indices_mask = np.where(edge_mask >= edge_mask_threshold[edge_type])[0]

    #         # Get the indices of the top 10 largest edge_mask values
    #         top_10_indices = heapq.nlargest(top_k, range(len(edge_mask)), edge_mask.take)
    #         edge_indices_mask = np.array(top_10_indices)

    #         # Assuming edge_indices_with_date and edge_indices are already defined
    #         set_indices_date = set(edge_indices_date)
    #         set_indices_mask = set(edge_indices_mask)

    #         # Union of the two sets
    #         merged_indices_set = set_indices_date.intersection(set_indices_mask)

    #         # Convert the merged set back into a NumPy array
    #         edge_indices = np.array(list(merged_indices_set))

    #     src_dst_pairs = [edge for edge, idx in edge_index_dicts[edge_type].items() if idx in edge_indices]
    #     thresholded_edges.update([(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs])

    # subgraph_nodes = G.subgraph(thresholded_nodes)


    # # # Add filtered nodes
    # # for n, d in subgraph_nodes.nodes(data=True):
    # #     subgraph.add_node(n, **d)

    # # Add filtered edges
    # for src, dst, edge_type in thresholded_edges:
    #     edge_data = G[src][dst][edge_type]
    #     subgraph.add_edge(src, dst, key=edge_type, **edge_data)

    #     # Add the src and dst nodes with their attributes
    #     src_node_data = G.nodes[src]
    #     dst_node_data = G.nodes[dst]
    #     subgraph.add_node(src, **src_node_data)
    #     subgraph.add_node(dst, **dst_node_data)
    
    

    # # Find the ticker's 1-hop 'naics' neighbor
    # ticker = target_nodes[1]
    # naics_neighbor = None

    # for neighbor, _ in G[ticker].items():
    #     if G.nodes[neighbor]['node_type'] == 'naics':
    #         naics_neighbor = neighbor
    #         break
    
    # if naics_neighbor is not None:
    #     # Add the naics neighbor node with its attributes
    #     subgraph.add_node(naics_neighbor, **G.nodes[naics_neighbor])

    #     # Add the edge between ticker and naics neighbor
    #     edge_data = G[ticker][naics_neighbor][('ticker', 'classified', 'naics')]
    #     subgraph.add_edge(ticker, naics_neighbor, key=('ticker', 'classified', 'naics'), **edge_data)

    # # # Add 1-hop "naics" neighbors to ticker nodes
    # # naics_neighbors_to_add = []

    # # for ticker_node in subgraph.nodes(data=True):
    # #     if ticker_node[1]['node_type'] == 'ticker':
    # #         for naics_neighbor, _ in G[ticker_node[0]].items():
    # #             if G.nodes[naics_neighbor]['node_type'] == 'naics':
    # #                 naics_neighbors_to_add.append((ticker_node[0], naics_neighbor))

    # # for ticker_node, naics_neighbor in naics_neighbors_to_add:
    # #     subgraph.add_node(naics_neighbor, **G.nodes[naics_neighbor])
    # #     edge_data = G[ticker_node][naics_neighbor][('ticker', 'classified', 'naics')]
    # #     subgraph.add_edge(ticker_node, naics_neighbor, key=('ticker', 'classified', 'naics'), **edge_data)

    # # return subgraph

node_colors = {
    'ticker': 'red',
    'congressperson': 'blue',
    'committee': 'green',
    'bill': 'orange',
    'naics': 'purple'
}

shapes = {
    'ticker': 'o',
    'congressperson': 's',
    'committee': '^',
    'bill': 'D',
    'naics': 'v'
}

# def find_specific_paths(graph, src, dst, node_type_sequence, target_ticker):
#     paths = []

#     def dfs(path, current_node_type_idx):
#         current_node = path[-1]
#         current_node_type = node_type_sequence[current_node_type_idx]

#         if current_node == dst and current_node_type_idx == len(node_type_sequence) - 1:
#             ticker_nodes = [node for node in path if graph.nodes[node]['node_type'] == 'ticker']
#             naics_nodes = [node for node in path if graph.nodes[node]['node_type'] == 'naics']

#             # Ensure that both ticker nodes share the same NAICS code
#             if naics_nodes[0] in graph.neighbors(ticker_nodes[0]) and naics_nodes[0] in graph.neighbors(ticker_nodes[1]):
#                 paths.append(path)

#             return

#         if current_node_type_idx >= len(node_type_sequence) - 1:
#             return

#         for neighbor in graph.neighbors(current_node):
#             if graph.nodes[neighbor]['node_type'] == node_type_sequence[current_node_type_idx + 1]:
#                 if current_node_type == 'ticker' and neighbor == target_ticker:
#                     continue
#                 dfs(path + [neighbor], current_node_type_idx + 1)

#     dfs([src], 0)
#     return paths

# read files in exp/trans_edge_not_included/results
import os
# folder_path = "exp/trans_edge_not_included/results_legacy"
# folder_path = "exp/trans_edge_not_included/results"
folder_path = "exp/trans_edge_not_included/results_wyden_klac"


pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

for pkl_file in pkl_files:
    print("pkl file: ", pkl_file)
    # with open("node_edge_masks_results.pkl", "rb") as f:
    with open(os.path.join(folder_path, pkl_file), "rb") as f:
        results = pickle.load(f)

    # extract the integer after "results_" in the filename
    congressperson_label = int(pkl_file.split("_")[7])
    ticker_label = int(pkl_file.split("_")[9])
    target_edge_attr = int(pkl_file.split("_")[10].replace("attr", '').split('.')[0])
    print("target edge attr: ", target_edge_attr)

    # target_edge_idx = edge_index_dicts[('congressperson', 'buy-sell', 'ticker')][(congressperson_label, ticker_label)]
    # target_edge_attr = data[('congressperson', 'buy-sell', 'ticker')].edge_attr[target_edge_idx]

    # make a reference date that we discard all the edges before then (particualry, all edges end date before then)
    # look_back = 365

    reference_date = target_edge_attr
    

    # Convert the reference date to a datetime object
    import datetime
    start_date = datetime.datetime(2016, 1, 1)
    ref_date = start_date + datetime.timedelta(days=reference_date)

    # Convert the datetime object to a string in the format "YYYY-xx-xx"
    ref_date_str = ref_date.strftime("%Y-%m-%d")

    print(congressperson_label, ticker_label)
    congressperson_id = reverse_congresspeople[congressperson_label] # id is semantic label
    ticker_id = reverse_tickers[ticker_label]

    # Read the node masks
    node_masks = results[(congressperson_label, ticker_label)]['node_masks']
    # Read the edge masks
    edge_masks = results[(congressperson_label, ticker_label)]['edge_masks']

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Concatenate node and edge mask values into a single array
    values = []
    labels = []
    # for key in node_masks:
    #     values.extend(node_masks[key])
    #     labels.extend([key] * len(node_masks[key]))
    # for key in edge_masks:
    #     values.extend(edge_masks[key])
    #     labels.extend([key] * len(edge_masks[key]))

    # # Create a density plot using Seaborn
    # sns.kdeplot(values, hue=labels, fill=True)

    # # Set plot title and axis labels
    # plt.title('Density Plot of Node and Edge Mask Values')
    # plt.xlabel('Mask Value')
    # plt.ylabel('Density')

    # plt.close()

    # Show the plot
    # plt.show()

    #### Check Sparsity ####
    import numpy as np

    # Assuming node_masks and edge_masks are the dictionaries with key-value pairs

    # Count the number of 0s and 1s in the node masks
    num_node_zeros = sum([np.count_nonzero(mask == 0) for mask in node_masks.values()])
    num_node_ones = sum([np.count_nonzero(mask == 1) for mask in node_masks.values()])

    # Count the number of 0s and 1s in the edge masks
    num_edge_zeros = sum([np.count_nonzero(mask == 0) for mask in edge_masks.values()])
    num_edge_ones = sum([np.count_nonzero(mask == 1) for mask in edge_masks.values()])

    # Count the number of non-zero and non-one values in the node masks
    num_node_nonzero = sum([np.count_nonzero((mask != 0) & (mask != 1)) for mask in node_masks.values()])

    # Count the number of non-zero and non-one values in the edge masks
    num_edge_nonzero = sum([np.count_nonzero((mask != 0) & (mask != 1)) for mask in edge_masks.values()])

    # Print the counts
    print("Number of 0s in node masks:", num_node_zeros)
    print("Number of 1s in node masks:", num_node_ones)
    print("Number of non-zero and non-one values in node masks:", num_node_nonzero)

    print("Number of 0s in edge masks:", num_edge_zeros)
    print("Number of 1s in edge masks:", num_edge_ones)
    print("Number of non-zero and non-one values in edge masks:", num_edge_nonzero)
    ####

    allow = 0.1
    allow_dicts_edges = {
        ('congressperson', 'buy-sell', 'ticker'): 0, 
        ('congressperson', 'assignment', 'committee'): 0, 
        ('ticker', 'lobbies_on', 'bill'): 0,
        ('bill', 'assigned_to', 'committee'): 1e-4, 
        ('ticker', 'classified', 'naics'): 0, 
        ('ticker', 'rev_buy-sell', 'congressperson'): 0, 
        ('committee', 'rev_assignment', 'congressperson'): 1e-5, 
        ('bill', 'rev_lobbies_on', 'ticker'): 1e-5, 
        ('committee', 'rev_assigned_to', 'bill'):1e-4, 
        ('naics', 'rev_classified', 'ticker'): 0
    }

    unit = 0.0125
    allow_dicts_edges = {
        ('congressperson', 'buy-sell', 'ticker'): 0, 
        ('congressperson', 'assignment', 'committee'): 0, 
        ('ticker', 'lobbies_on', 'bill'): unit,
        ('bill', 'assigned_to', 'committee'): 0, 
        ('ticker', 'classified', 'naics'): 0, 
        ('ticker', 'rev_buy-sell', 'congressperson'): 0, 
        ('committee', 'rev_assignment', 'congressperson'): 0, 
        ('bill', 'rev_lobbies_on', 'ticker'): unit, 
        ('committee', 'rev_assigned_to', 'bill'):0, 
        ('naics', 'rev_classified', 'ticker'): 0
    }

    # Find the maximum value for each type in node_masks and store them in a dict
    max_node_masks = {key: max(value) - allow for key, value in node_masks.items()}

    print(max_node_masks.keys())

    # Find the maximum value for each type in edge_masks and store them in a dict
    max_edge_masks = {key: max(value) - allow_dicts_edges[key] for key, value in edge_masks.items()}

    # # Add the following inside the loop that iterates over results.keys()
    # node_mask_threshold = 0.999
    # edge_mask_threshold = 0.999

    congressperson = integer_to_semantic_index['congressperson'][congressperson_label]
    ticker = integer_to_semantic_index['ticker'][ticker_label]

    print(f"Congressperson: {congressperson}")
    print(f"Ticker: {ticker}")
    print(f"Reference date: {ref_date_str}")

    target_nodes = [congressperson, ticker]

    subgraph = get_thresholded_subgraph(loaded_G, node_masks, edge_masks, max_node_masks, max_edge_masks, semantic_to_integer_index, edge_types, edge_index_dicts, target_nodes, reference_date)

    def find_paths(subgraph, target_ticker, target_congressperson):
        paths = []
    
        def dfs(path, current_node, target_node, depth):
            if depth == 0:
                if current_node == target_node:
                    paths.append(path)
                return
            for neighbor, edge_data in subgraph[current_node].items():
                for edge_type, _ in edge_data.items():
                    if depth == 4:
                        if edge_type == ('bill', 'assigned_to', 'committee'):
                            dfs(path + [neighbor], neighbor, target_node, depth - 1)
                    elif depth == 6:
                        if edge_type == ('ticker', 'lobbies_on', 'bill'):
                            dfs(path + [neighbor], neighbor, target_node, depth - 1)
                        elif edge_type == ('ticker', 'classified', 'naics'):
                            dfs(path + [neighbor], neighbor, target_node, depth - 1)
                    else:
                        dfs(path + [neighbor], neighbor, target_node, depth - 1)
        
        for neighbor, edge_data in subgraph[target_ticker].items():
            for edge_type, _ in edge_data.items():
                if edge_type == ('ticker', 'lobbies_on', 'bill'):
                    dfs([target_ticker, neighbor], neighbor, target_congressperson, 4)
                elif edge_type == ('ticker', 'classified', 'naics'):
                    dfs([target_ticker, neighbor], neighbor, target_congressperson, 6)
        
        return paths

    # Example usage
    paths = find_paths(subgraph, target_nodes[1], target_nodes[0])
    print("Number of paths:", len(paths))
    # Print the paths
    for path in paths:
        print("path", path)

    pass
    # node_type_sequence = ["congressperson", "committee", "bill", "ticker", "naics", "ticker", "bill", "committee", "congressperson"]

    # specific_paths = find_specific_paths(subgraph, target_nodes[0], target_nodes[0], node_type_sequence, target_nodes[1])

    # subgraph = nx.subgraph(subgraph, set().union(*specific_paths))

    # Get connected components with the target ticker node included
    # connected_components = [comp for comp in nx.connected_components(subgraph) if (target_nodes[1] in comp or target_nodes[0] in comp)]

    # Combine the connected components with the target ticker node to form the final subgraph
    # nodes_to_include = set().union(*connected_components)
    # subgraph = subgraph.subgraph(nodes_to_include)
    # # Combine the connected components with the target ticker node to form the final subgraph
    # nodes_to_include = set().union(*connected_components)
    # subgraph = subgraph.subgraph(nodes_to_include)

    # # Get the ego graphs for both target nodes with a maximum radius of 3
    # ego_graph_1 = nx.ego_graph(subgraph, target_nodes[0], radius=3)
    # ego_graph_2 = nx.ego_graph(subgraph, target_nodes[1], radius=3)

    # # Combine the two ego graphs
    # combined_ego_graph = nx.compose(ego_graph_1, ego_graph_2)

    # # Assign the combined ego graph as the final subgraph
    # subgraph = combined_ego_graph

    # # # Find all simple cycles in the subgraph
    # subgraph_directed = nx.DiGraph(subgraph)
    # cycles = list(nx.simple_cycles(subgraph_directed))

    # print(f"Number of cycles: {len(cycles)}")
    # # Filter cycles that include both target_nodes[0] and target_nodes[1]
    # filtered_cycles = [cycle for cycle in cycles if (target_nodes[0] in cycle and target_nodes[1] in cycle)]
    # print(f"Number of filtered cycles: {len(filtered_cycles)}")

    # # Combine the filtered cycles to form the final subgraph
    # nodes_to_include = set().union(*filtered_cycles)
    # subgraph = subgraph.subgraph(nodes_to_include)


    # # find largest connected component
    # largest_connected_comp = max(nx.connected_components(subgraph), key=len)
    # subgraph = subgraph.subgraph(largest_connected_comp)

    # # Get connected components with at least 3 nodes
    # connected_components = [comp for comp in nx.connected_components(subgraph) if (len(comp)>=1)]

    # # Combine the connected components to form the final subgraph
    # nodes_to_include = set().union(*connected_components)
    # subgraph = subgraph.subgraph(nodes_to_include)

    title = f"Subgraph for {congressperson} and {ticker} on {ref_date_str}"
    draw_subgraph(subgraph, node_colors, shapes, title=title, congressperson_label=congressperson, ticker_label=ticker)
    pass

    pass

