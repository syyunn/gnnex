import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, MessagePassing
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

# model = CustomGNN(num_nodes_dict=num_nodes_dict, embedding_dim=64, edge_dim=32, num_edge_features=1, out_channels=64)
class CustomGNN(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, edge_dim, num_edge_features, out_channels):
        super(CustomGNN, self).__init__()

        # Separate embeddings for different node types
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, embedding_dim)
            for node_type, num_nodes in num_nodes_dict.items()
        })

        # Node-type-specific weight matrices
        self.node_transforms = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(embedding_dim, embedding_dim)
            for node_type in num_nodes_dict.keys()
        })

        # Define the NN for NNConv
        nn_layer = nn.Sequential(
            nn.Linear(num_edge_features, embedding_dim * out_channels),
            nn.ReLU(),
            nn.Linear(embedding_dim * out_channels, embedding_dim * out_channels)
        )

        # Define NNConv layer
        self.conv = NNConv(embedding_dim, out_channels, nn_layer, aggr='mean')

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, x_dict, edge_index, edge_attr):
        # Apply separate embeddings and node-type-specific weight matrices
        x_dict = {node_type: self.node_transforms[node_type](self.embeddings[node_type](x))
                  for node_type, x in x_dict.items()}

        # Concatenate node embeddings
        x = torch.cat(list(x_dict.values()), dim=0)

        # Apply NNConv
        x = self.conv(x, edge_index, edge_attr)
        x = F.relu(x)

        # Output layer
        out = self.output_layer(x)
        return out

# Define the number of nodes for each node type
num_nodes_dict = {'congressperson': 2431, 'committee': 556, 'ticker': 4202, 'bill': 47767, 'naics': 744}

# Instantiate the model
model = CustomGNN(num_nodes_dict=num_nodes_dict, embedding_dim=64, edge_dim=32, num_edge_features=1, out_channels=64)

# Forward pass (example)
x_dict = {'congressperson': torch.randint(2431, (10,)), 'committee': torch.randint(556, (10,))}
edge_index = torch.randint(10, (2, 20))
edge_attr = torch.rand(20, 1)
output = model(x_dict, edge_index, edge_attr)
print(output.shape)
