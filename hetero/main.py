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

# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    
# data = data.to(device)

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
print("edge_label_index", edge_label_index)
edge_label = train_data["congressperson", "buy-sell", "ticker"].edge_label
print("edge_label", edge_label)

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("congressperson", "buy-sell", "ticker"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

# Define the model
from model import CustomGNN

# Given the HeteroData object named 'data'
num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}

# Print the num_nodes_dict
print(num_nodes_dict)

# Instantiate the model
model = CustomGNN(num_nodes_dict=num_nodes_dict, embedding_dim=64, edge_dim=32, num_edge_features=1, out_channels=64)

# Define the loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Extract node IDs for all node types
        x_dict = {key: batch[key].node_id for key in data.node_types}

        preds = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch["congressperson", "buy-sell", "ticker"].edge_label_index)
        labels = batch["congressperson", "buy-sell", "ticker"].edge_label

        loss = criterion(preds, labels)

        # Update the model parameters
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")
