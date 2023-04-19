## This code is deprecated but preserved for future reference only.

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from hetero_gnnex import HeteroGNNExplainer
from model import BuySellLinkPrediction
from model_no_embeddings import BuySellLinkPredictionNoEmbedding

import pickle


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
    edge_types=edge_types,
    num_layers=num_layers,
).to(device)

# Load the model state
model_path = "buysell_link_prediction_best_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device))

### Get target as preds
# Evaluate the model
model.eval()

# congressperson_id, ticker_id = data[('congressperson', 'buy-sell', 'ticker')]['edge_index']
# edge_label_index = torch.stack([congressperson_id, ticker_id], dim=0)

edge_label_index = data[('congressperson', 'buy-sell', 'ticker')]['edge_index']
edge_label_attr = data[('congressperson', 'buy-sell', 'ticker')]['edge_attr']

# date scaling
from datetime import date

start_date = date(2016, 1, 1)
today = date.today()
total_days = (today - start_date).days

# Create edge_label_attr tensor
edge_label_attr = edge_label_attr / total_days
print("Edge label attribute:", edge_label_attr)
# Prepare the input data for the model

x_dict = {node_type: data[node_type].node_id for node_type in num_nodes_dict.keys()}
scaled_edge_attr_dict = {key: value / total_days for key, value in data.edge_attr_dict.items()}

# Perform inference using the trained model
with torch.no_grad():
    # preds = model(x_dict, batch.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=batch_edge_label_attr)
    preds = model(x_dict, data.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=edge_label_attr)

# Print the prediction results
print("Prediction result:", preds)

target = preds

# Extract the model's embeddings
embedding_dict = model.gnn.embeddings

# Instantiate the new model without the embedding layer
model_no_embedding = BuySellLinkPredictionNoEmbedding(
    num_nodes_dict,
    embedding_dim=embedding_dim,
    num_edge_features=num_edge_features,
    out_channels=out_channels,
    edge_types=edge_types,
    num_layers=num_layers,
).to(device)

# Load the state dict of the model with embeddings
state_dict_with_embeddings = torch.load(model_path, map_location=device)

# Remove the embeddings from the state dict
state_dict_no_embeddings = {key: value for key, value in state_dict_with_embeddings.items()
                            if not key.startswith('gnn.embeddings')}

# Load the state dict into the model without embeddings
model_no_embedding.load_state_dict(state_dict_no_embeddings)


# Instantiate the HeteroGNNExplainer
explainer = HeteroGNNExplainer(model=model_no_embedding, epochs=1, lr=0.01, device=device, data=data, edge_label_index=edge_label_index, edge_label_attr=edge_label_attr)

# Prepare the edge of interest
which_edge = 0
congressperson_id, ticker_id = data[('congressperson', 'buy-sell', 'ticker')]['edge_index'][:, which_edge]

edge_to_explain = torch.tensor([congressperson_id, ticker_id], device=device)  # Replace with your edge of interest
edge_type_to_explain = ("congressperson", "buy-sell", "ticker")


# Run the explain_edge method
node_masks, edge_masks = explainer(
    model = model_no_embedding,
    x_dict = embedding_dict,
    edge_index_dict = data.edge_index_dict,
    target = target,
    index = None
)

print("Node masks:", node_masks)
print("Node masks:", node_masks.shape)
print("Edge masks:", edge_masks)
print("Edge masks:", edge_masks.shape)
