import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from hetero_gnnex import HeteroGNNExplainer
from model import BuySellLinkPrediction
from model_no_embeddings import BuySellLinkPredictionNoEmbedding

import pickle

from tqdm import tqdm

# Load your HeteroData object named 'data'
with open("./data/hetero_graph_data.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Extract the data from the loaded dictionary
data = loaded_data["hetero_graph"]
unique_tickers = loaded_data["unique_tickers"]
unique_congresspeople = loaded_data["unique_congresspeople"]
unique_committees = loaded_data["unique_committees"]
unique_bills = loaded_data["unique_bills"]
unique_naics = loaded_data["unique_naics"]

# Create reverse dictionaries
reverse_data = {v: k for k, v in data.items()}
reverse_unique_tickers = {v: k for k, v in unique_tickers.items()}
reverse_unique_congresspeople = {v: k for k, v in unique_congresspeople.items()}
reverse_unique_committees = {v: k for k, v in unique_committees.items()}
reverse_unique_bills = {v: k for k, v in unique_bills.items()}
reverse_unique_naics = {v: k for k, v in unique_naics.items()}

import torch

# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))

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

# Collect edge_types 
edge_types = []
# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    edge_types.append(edge_type)

print("Edge types:", edge_types)
print(len(edge_types))

model_edge_types = [edge_type for edge_type in edge_types if edge_type not in [("congressperson", "buy-sell", "ticker"), ("ticker", "rev_buy-sell", "congressperson")]]

# Given the HeteroData object named 'data'
num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}
# Print the num_nodes_dict
print(num_nodes_dict)

# Prepare data and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# Define hyperparameters
num_layers = 2
embedding_dim = 64
num_edge_features = 2
out_channels = 64

# Instantiate the model
model = BuySellLinkPrediction(
    num_nodes_dict,
    embedding_dim=embedding_dim,
    num_edge_features=num_edge_features,
    out_channels=out_channels,
    # edge_types=edge_types,
    edge_types=model_edge_types,
    num_layers=num_layers,
).to(device)

# Load the model state
# model_path = "buysell_link_prediction_best_model.pt"
model_path = "buysell_link_prediction_best_model_accu_0.9889400921658986_new.pt"
model.load_state_dict(torch.load(model_path, map_location=device))

### Get target as preds
# Evaluate the model
model.eval()

# Extract the model's embeddings
embedding_dict = model.gnn.embeddings

# Instantiate the new model without the embedding layer
model_no_embedding = BuySellLinkPredictionNoEmbedding(
    num_nodes_dict,
    embedding_dim=embedding_dim,
    num_edge_features=num_edge_features,
    out_channels=out_channels,
    # edge_types=edge_types,
    edge_types=model_edge_types,
    num_layers=num_layers,
).to(device)

# Load the state dict of the model with embeddings
state_dict_with_embeddings = torch.load(model_path, map_location=device)

# Remove the embeddings from the state dict
state_dict_no_embeddings = {key: value for key, value in state_dict_with_embeddings.items()
                            if not key.startswith('gnn.embeddings')}

# Load the state dict into the model without embeddings
model_no_embedding.load_state_dict(state_dict_no_embeddings)

# Find specific edge index
edge_indices = data[('congressperson', 'buy-sell', 'ticker')]['edge_index']

src_indices = edge_indices[0, :]
dest_indices = edge_indices[1, :]

src = torch.tensor(2361)
dest = torch.tensor(234)
# Find the index of the pair (2361, 234)
import numpy as np
indices = np.where((src_indices.cpu() == src) & (dest_indices.cpu() == dest))


# Instantiate the HeteroGNNExplainer
epochs = 200
lr = 10
l1_lambda = 1000
custom_lambda = 0.01

# Prepare the edge of interest (whole)
which_edges = [i for i in range(data[('congressperson', 'buy-sell', 'ticker')]['edge_index'].shape[1])]

# Prepare the edge of interest (specific)
which_edges = list(indices[0])

# already_done = 0
import os
import re
import numpy as np

results_directory = "exp/results"
result_files = os.listdir(results_directory)

# Extract idx values from the filenames
idx_values = [int(re.findall(r'\d+', file)[0]) for file in result_files if "node_edge_masks_results" in file]

if idx_values:
    idx_values.sort()
    for i, val in enumerate(idx_values):
        if i != val:
            already_done = i
            break
    else:
        already_done = max(idx_values) + 1
else:
    already_done = 0

print(f"Starting from idx: {already_done}")

dates = [i for i in range(-846, 2779)]

for d in dates:
    results = {}
    congressperson_id, ticker_id = src, dest
    edge_label_index = torch.stack([congressperson_id, ticker_id], dim=0)
    print("Congressperson ID: ", congressperson_id)
    print("Ticker ID: ", ticker_id)
    print("Edge label index: ", edge_label_index)

    # date scaling
    from datetime import date

    start_date = date(2016, 1, 1)
    today = date.today()
    total_days = (today - start_date).days

    print("date", d)
    d = 100000000
    edge_attr_before_scaled = torch.tensor([d, d])
    raw_attr = edge_attr_before_scaled[0]
    print("Raw attr: ", raw_attr)

    edge_attr = torch.tensor(edge_attr_before_scaled/total_days, dtype=torch.float, device=device)
    print("Edge label attribute: ", edge_attr)

    # Prepare the input data for the model
    x_dict = {node_type: data[node_type].node_id for node_type in num_nodes_dict.keys()}

    scaled_edge_attr_dict = {key: value / total_days for key, value in data.edge_attr_dict.items()}

    # Perform inference using the trained model
    with torch.no_grad():
        # preds = model(x_dict, batch.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=batch_edge_label_attr)
        preds, preds_before_sig = model(x_dict, data.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=edge_attr)
    
    print("Preds: ", preds)
    print("Preds before sig: ", preds_before_sig)

