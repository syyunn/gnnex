import pickle
from tqdm import tqdm

import numpy as np

import random

import torch

# # Set the random seed for reproducibility
# seed = 17806
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# Open the file in binary read mode and unpickle the data
with open('/home/gridsan/syun/gnnex/data/hetero_graph_data.pkl', "rb") as f:
# with open('./data/hetero_graph_data.pkl', "rb") as f:
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

import torch_geometric.transforms as T

# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:

transform = T.RandomLinkSplit(
    num_val=0,
    num_test=0.1,
    disjoint_train_ratio=0.3, # Across the training edges, we use 70% of edges for message passing, and 30% of edges for supervision.
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=("congressperson", "buy-sell", "ticker"),
    rev_edge_types=("ticker", "rev_buy-sell", "congressperson"), 
)
train_data, val_data, test_data = transform(data)

# Check unique values after applying the transform
transformed_edge_label = train_data["congressperson", "buy-sell", "ticker"].edge_label

# split the data into train and test

from torch_geometric.loader import LinkNeighborLoader

#   Define seed edges:
edge_label_index = train_data["congressperson", "buy-sell", "ticker"].edge_label_index
edge_label = train_data["congressperson", "buy-sell", "ticker"].edge_label
edge_attr = train_data["congressperson", "buy-sell", "ticker"].edge_attr

# Create a dictionary to map edge indices to their attributes
edge_to_attr = {(src.item(), dst.item()): attr.to(device) for src, dst, attr in zip(*edge_label_index, edge_attr)}


# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:

num_neigbors = [20, 10, 5]
batch_size = 128
print("batch_size", batch_size)

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=num_neigbors,
    edge_label_index=(("congressperson", "buy-sell", "ticker"), edge_label_index),
    edge_label=edge_label,
    batch_size=batch_size,
    shuffle=True,
)

# Define seed edges for the test dataset:
test_edge_label_index = test_data["congressperson", "buy-sell", "ticker"].edge_label_index
test_edge_label = test_data["congressperson", "buy-sell", "ticker"].edge_label
test_edge_attr = test_data["congressperson", "buy-sell", "ticker"].edge_attr

print("test_edge_label_index", test_edge_label_index)
print("test_edge_label", test_edge_label)
# Create a dictionary to map edge indices to their attributes
test_edge_to_attr = {(src.item(), dst.item()): attr.to(device) for src, dst, attr in zip(*test_edge_label_index, test_edge_attr)}


# Create the test loader:
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=num_neigbors,  # Same number of neighbors as in the training loader
    edge_label_index=(("congressperson", "buy-sell", "ticker"), test_edge_label_index),
    edge_label=test_edge_label,
    batch_size=batch_size,  # Same batch size as in the training loader
    shuffle=False,  # No need to shuffle the test dataset
)

# Define the model
from model import BuySellLinkPrediction

# Given the HeteroData object named 'data'
num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}

# Print the num_nodes_dict
print(num_nodes_dict)

# Instantiate the model
num_layers = 2
print("num_layers", num_layers)
model = BuySellLinkPrediction(num_nodes_dict, embedding_dim=64, num_edge_features=2, out_channels=64, edge_types=edge_types, num_layers=num_layers).to(device)

# Training loop
import torch.optim as optim
from torch.nn import functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from torch.optim.lr_scheduler import StepLR

epochs = 100
optimizer = optim.Adam(model.parameters(), lr=0.005)  # You can set the learning rate (lr) as needed

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decay the learning rate by a factor of 0.1 every 10 epochs

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_auc_roc = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device) # Move the batch to the device
        batch_edge_label = batch[("congressperson", "buy-sell", "ticker")].edge_label

        optimizer.zero_grad()
        
        # Get node embeddings from the batch
        x_dict = {node_type: batch[node_type].node_id for node_type in num_nodes_dict.keys()}
        
        # Get edge_label_index and edge_label
        edge_label_index = batch[("congressperson", "buy-sell", "ticker")].edge_label_index
        edge_label = batch[("congressperson", "buy-sell", "ticker")].edge_label

        from util import get_edge_attr_for_batch
        batch_edge_label_attr = get_edge_attr_for_batch(train_data["congressperson", "buy-sell", "ticker"].edge_index, train_data["congressperson", "buy-sell", "ticker"].edge_attr, edge_label_index, edge_to_attr)
        batch_edge_label_attr = batch_edge_label_attr.to(device)

        # Count the number of samples with label 0 (negative samples)
        num_negatives = torch.sum(edge_label == 0).item()

        # Count the number of samples with label 1 (positive samples)
        num_positives = torch.sum(edge_label == 1).item()

        # # Print the counts
        # print(f"Number of negative samples (label 0): {num_negatives}")
        # print(f"Number of positive samples (label 1): {num_positives}")
        
        # print("batch.edge_attr_dict", batch.edge_attr_dict)

        # date scaling
        from datetime import date

        start_date = date(2016, 1, 1)
        today = date.today()

        total_days = (today - start_date).days

        scaled_edge_attr_dict = {
            key: value / total_days
            for key, value in batch.edge_attr_dict.items()
        }
        # print("scaled_edge_attr_dict", scaled_edge_attr_dict)
        # Forward pass
        preds = model(x_dict, batch.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=batch_edge_label_attr)
        # print("preds", preds)
        
        # Compute loss
        loss = F.binary_cross_entropy(preds, edge_label.float())
        # print("label", edge_label)
        total_loss += loss.item()

         # Convert predicted probabilities to binary predictions
        binary_preds = (preds > 0.5).float()
        # print("binary_preds", binary_preds)
        
        # Compute accuracy
        accuracy = accuracy_score(edge_label.cpu().numpy(), binary_preds.cpu().numpy())
        total_accuracy += accuracy

        # Compute AUC-ROC
        auc_roc = roc_auc_score(edge_label.cpu().detach().numpy(), preds.cpu().detach().numpy())      
        total_auc_roc += auc_roc  

        # Backward pass
        loss.backward()
        optimizer.step()

    # Update the learning rate using the scheduler
    scheduler.step()

    # Print loss
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_auc_roc = total_auc_roc / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Train Accuracy: {avg_accuracy:.4f} Train AUC-ROC: {avg_auc_roc:.4f}")

    # eval
    # Evaluate the model on the test dataset
    from util import evaluate
    test_loss, test_accuracy, test_auc_roc = evaluate(test_loader, model, device, num_nodes_dict, test_data, test_edge_to_attr)
    print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f} Test AUC-ROC: {test_auc_roc:.4f}")

# Save the model after the training loop
torch.save(model.state_dict(), "buysell_link_prediction_model.pt")