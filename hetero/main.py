import pickle


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


print("Data has been loaded from the pickle file.")
print(data)

import torch

# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    
data = data.to(device)

import torch_geometric.transforms as T

# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("congressperson", "buy-sell", "ticker"),
    rev_edge_types=("ticker", "rev_buy-sell", "congressperson"), 
)
train_data, val_data, test_data = transform(data)


# split the data into train and test

# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:
from torch_geometric.loader import LinkNeighborLoader

# Define seed edges:
edge_label_index = train_data["congressperson", "buy-sell", "ticker"].edge_label_index
print(edge_label_index)
edge_label = train_data["congressperson", "buy-sell", "ticker"].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("congressperson", "buy-sell", "ticker"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MetaLayer, global_add_pool
from torch_geometric.nn import SAGEConv, to_hetero
from typing import Dict


from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_dict_new = {}
        for edge_type_key in data.edge_types:
            src_key, edge_index_key, dst_key = edge_type_key
            x_src = x_dict[src_key]
            
            # Get the correct edge_index for the current edge type
            edge_index = edge_index_dict[edge_type_key]
            
            # Use only x_src and edge_index as inputs to SAGEConv
            x_dst_new = F.relu(self.conv1(x_src, edge_index))
            x_dst_new = self.conv2(x_src, edge_index)
            x_dict_new[dst_key] = x_dst_new
        return x_dict_new


class Classifier(torch.nn.Module):
    def forward(self, x_congressperson, x_ticker, edge_label_index):
        edge_feat_congressperson = x_congressperson[edge_label_index[0]]
        edge_feat_ticker = x_ticker[edge_label_index[1]]
        return (edge_feat_congressperson * edge_feat_ticker).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, data: HeteroData, hidden_channels: int):
        super().__init__()
        self.congressperson_emb = torch.nn.Embedding(data['congressperson'].num_nodes, hidden_channels)
        self.ticker_emb = torch.nn.Embedding(data['ticker'].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_congressperson = self.congressperson_emb(data['congressperson'].node_id)
        x_ticker = self.ticker_emb(data['ticker'].node_id)
        x_dict = {'congressperson': x_congressperson, 'ticker': x_ticker}
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(x_dict['congressperson'], x_dict['ticker'],
                               data['congressperson', 'buy-sell', 'ticker'].edge_label_index)
        return pred

model = Model(data, hidden_channels=64)
model = model.to(device)


import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_examples = 0
    for batch in train_loader:
        # Move batch to the appropriate device
        batch = batch.to(device)

        # Print data type of all edge index tensors
        for edge_type, edge_index in data.edge_index_dict.items():
            print(f"Data type of edge_index for edge type {edge_type}: {edge_index.dtype}")

        
        # Forward pass
        link_logits = model(batch)
        link_labels = batch.edge_label.float()
        
        # Compute loss
        loss = criterion(link_logits, link_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update loss and example count
        total_loss += loss.item() * batch.num_edges
        total_examples += batch.num_edges

    # Compute average loss
    avg_loss = total_loss / total_examples

    # Evaluate model on validation set
    model.eval()
    with torch.no_grad():
        val_logits = model(val_data).squeeze()
        val_labels = val_data.edge_label.float()
        auc = roc_auc_score(val_labels.cpu(), val_logits.cpu())

    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, AUC: {auc:.4f}')

print('Training complete.')
