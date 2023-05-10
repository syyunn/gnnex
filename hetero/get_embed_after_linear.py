import torch
from model import BuySellLinkPrediction
import pickle
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

import numpy as np

# Given the HeteroData object named 'data'
num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}

# Define hyperparameters
num_layers = 2
embedding_dim = 64
num_edge_features = 2
out_channels = 64

# Collect edge_types 
edge_types = []
# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    edge_types.append(edge_type)

model_edge_types = [edge_type for edge_type in edge_types if edge_type not in [("congressperson", "buy-sell", "ticker"), ("ticker", "rev_buy-sell", "congressperson")]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Step 1: Create TSV file with the embeddings
with open("embeddings.tsv", "w") as embeddings_file:
    for category in embedding_dict.keys():
        embedding_layer = model.gnn.embeddings[category]
        linear_layer = model.gnn.node_type_linear[category]

        num_embeddings = embedding_layer.weight.shape[0]
        for idx in range(num_embeddings):
            embedding = embedding_layer(torch.tensor(idx)).unsqueeze(0)
            initial_embedding = linear_layer(embedding)
            initial_embedding = initial_embedding.detach().cpu().numpy()

            embeddings_file.write("\t".join([str(x) for x in initial_embedding[0]]))
            embeddings_file.write("\n")

# Step 2: Create TSV file with metadata
with open("metadata.tsv", "w") as metadata_file:
    # Write header
    metadata_file.write("Type\tLabel\n")
    
    # Write metadata for each category
    for category in embedding_dict.keys():
        if category == "congressperson":
            reverse_dict = reverse_unique_congresspeople
        elif category == "committee":
            reverse_dict = reverse_unique_committees
        elif category == "ticker":
            reverse_dict = reverse_unique_tickers
        elif category == "bill":
            reverse_dict = reverse_unique_bills
        elif category == "naics":
            reverse_dict = reverse_unique_naics
        
        num_embeddings = embedding_dict[category].weight.shape[0]
        for idx in range(num_embeddings):
            metadata_file.write(f"{category}\t{reverse_dict[idx]}\n")
