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

# with open("node_edge_masks_results.pkl", "rb") as f:
with open("exp/trans_edge_not_included/node_edge_masks_results_8_0.1_new_new.pkl", "rb") as f:
    results = pickle.load(f)


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
    

    pos = nx.spring_layout(subgraph, seed=42, k=0.35)
    
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
    thresholded_nodes = set()
    thresholded_edges = set()

    for node_type, node_mask in node_masks.items():
        node_indices = np.where(node_mask >= node_mask_threshold[node_type])[0]
        thresholded_nodes.update([integer_to_semantic_index[node_type][i] for i in node_indices])
    
    # # Add target nodes to the set
    # for node_label in target_nodes:
    #     thresholded_nodes.add(node_label)

    for edge_type, edge_mask in edge_masks.items():
        edge_attr = data[edge_type].edge_attr

        if edge_type == ('ticker', 'classified', 'naics') or edge_type == ('naics', 'rev_classified', 'ticker'):
            edge_indices_mask = np.where(edge_mask > edge_mask_threshold[edge_type])[0]
            edge_indices = edge_indices_mask
        else:              
            # edge_indices_date = np.where(edge_attr[:, 0] < reference_date)[0] # start date should be "earlier" than reference date
            edge_indices_date = np.where((edge_attr[:, 0] >= (reference_date - 365)) & (edge_attr[:, 0] < reference_date))[0]
            edge_indices_mask = np.where(edge_mask >= edge_mask_threshold[edge_type])[0]

            # Assuming edge_indices_with_date and edge_indices are already defined
            set_indices_date = set(edge_indices_date)
            set_indices_mask = set(edge_indices_mask)

            # Union of the two sets
            merged_indices_set = set_indices_date.intersection(set_indices_mask)

            # Convert the merged set back into a NumPy array
            edge_indices = np.array(list(merged_indices_set))

        src_dst_pairs = [edge for edge, idx in edge_index_dicts[edge_type].items() if idx in edge_indices]
        thresholded_edges.update([(integer_to_semantic_index[edge_type[0]][src], integer_to_semantic_index[edge_type[2]][dst], edge_type) for src, dst in src_dst_pairs])

    subgraph_nodes = G.subgraph(thresholded_nodes)

    # Create a new MultiGraph to store the filtered subgraph
    subgraph = nx.MultiGraph()

    # # Add filtered nodes
    # for n, d in subgraph_nodes.nodes(data=True):
    #     subgraph.add_node(n, **d)

    # Add filtered edges
    for src, dst, edge_type in thresholded_edges:
        edge_data = G[src][dst][edge_type]
        subgraph.add_edge(src, dst, key=edge_type, **edge_data)

        # Add the src and dst nodes with their attributes
        src_node_data = G.nodes[src]
        dst_node_data = G.nodes[dst]
        subgraph.add_node(src, **src_node_data)
        subgraph.add_node(dst, **dst_node_data)
    
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
        # Add the naics neighbor node with its attributes
        subgraph.add_node(naics_neighbor, **G.nodes[naics_neighbor])

        # Add the edge between ticker and naics neighbor
        edge_data = G[ticker][naics_neighbor][('ticker', 'classified', 'naics')]
        subgraph.add_edge(ticker, naics_neighbor, key=('ticker', 'classified', 'naics'), **edge_data)

    # Add 1-hop "naics" neighbors to ticker nodes
    naics_neighbors_to_add = []

    for ticker_node in subgraph.nodes(data=True):
        if ticker_node[1]['node_type'] == 'ticker':
            for naics_neighbor, _ in G[ticker_node[0]].items():
                if G.nodes[naics_neighbor]['node_type'] == 'naics':
                    naics_neighbors_to_add.append((ticker_node[0], naics_neighbor))

    for ticker_node, naics_neighbor in naics_neighbors_to_add:
        subgraph.add_node(naics_neighbor, **G.nodes[naics_neighbor])
        edge_data = G[ticker_node][naics_neighbor][('ticker', 'classified', 'naics')]
        subgraph.add_edge(ticker_node, naics_neighbor, key=('ticker', 'classified', 'naics'), **edge_data)

    return subgraph

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


for congressperson_label, ticker_label in results.keys():

    target_edge_idx = edge_index_dicts[('congressperson', 'buy-sell', 'ticker')][(congressperson_label, ticker_label)]
    target_edge_attr = data[('congressperson', 'buy-sell', 'ticker')].edge_attr[target_edge_idx]

    print("transaction date: ", target_edge_attr[0])

    # make a reference date that we discard all the edges before then (particualry, all edges end date before then)
    # look_back = 365
    reference_date = target_edge_attr[0]

    # Convert the reference date to a datetime object
    import datetime
    start_date = datetime.datetime(2016, 1, 1)
    ref_date = start_date + datetime.timedelta(days=reference_date.item())

    # Convert the datetime object to a string in the format "YYYY-xx-xx"
    ref_date_str = ref_date.strftime("%Y-%m-%d")

    print(congressperson_label, ticker_label)
    congressperson_id = reverse_congresspeople[congressperson_label] # id is semantic label
    ticker_id = reverse_tickers[ticker_label]

    # Read the node masks
    node_masks = results[(congressperson_label, ticker_label)]['node_masks']
    # Read the edge masks
    edge_masks = results[(congressperson_label, ticker_label)]['edge_masks']


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

    allow = 1

    # Find the maximum value for each type in node_masks and store them in a dict
    max_node_masks = {key: max(value)*allow for key, value in node_masks.items()}

    # Find the maximum value for each type in edge_masks and store them in a dict
    max_edge_masks = {key: max(value)*allow for key, value in edge_masks.items()}

    # # Add the following inside the loop that iterates over results.keys()
    # node_mask_threshold = 0.999
    # edge_mask_threshold = 0.999

    congressperson = integer_to_semantic_index['congressperson'][congressperson_label]
    ticker = integer_to_semantic_index['ticker'][ticker_label]

    print(f"Congressperson: {congressperson}")
    print(f"Ticker: {ticker}")

    target_nodes = [congressperson, ticker]

    subgraph = get_thresholded_subgraph(loaded_G, node_masks, edge_masks, max_node_masks, max_edge_masks, semantic_to_integer_index, edge_types, edge_index_dicts, target_nodes, reference_date)

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

