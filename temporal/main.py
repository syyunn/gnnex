import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data

# Load and preprocess the dataset
dataset = KarateClub()
data = dataset[0]
data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
data = ToUndirected()(data)

# Generate random temporal edge features
temporal_edge_features = torch.rand((data.train_pos_edge_index.size(1), 1))

class TemporalGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Initialize the GNN model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalGNN(dataset.num_features, hidden_channels=16, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

data = data.to(device)
temporal_edge_features = temporal_edge_features.to(device)
x, edge_index, edge_attr = data.x, data.train_pos_edge_index, temporal_edge_features

def train():
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_attr)
    loss = F.cross_entropy(out[data.train_pos_edge_index[:, 0]], data.y[data.train_pos_edge_index[:, 0]])
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    loss = train()
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Inference example
with torch.no_grad():
    model.eval()
    new_temporal_edge_feature = torch.tensor([[0.5]])
    new_edge_index = torch.tensor([[0], [33]], dtype=torch.long).to(device)
    print(new_edge_index)
    print(new_edge_index)
    out = model(x, new_edge_index, new_temporal_edge_feature)
    prob = F.softmax(out[new_edge_index[:, 0]], dim=1)[:, 1]
    print(f"Probability of having an edge between nodes {new_edge_index[:, 0].tolist()} with temporal edge feature {new_temporal_edge_feature.item()}: {prob[0].item():.4f}")
