import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import NNConv
from torch_geometric.nn import HeteroConv
from debug import NNConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers=2):
        super(HeteroGNN, self).__init__()

        # Separate embeddings for different node types
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, embedding_dim)
            for node_type, num_nodes in num_nodes_dict.items()
        })

        # Define separate linear layers for each node type
        self.node_type_linear = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(embedding_dim, embedding_dim)
            for node_type in num_nodes_dict.keys()
        })

        # Define separate NNs for each edge type
        nn_dict = { 
            edge_type: nn.Sequential(
                nn.Linear(num_edge_features, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, embedding_dim*out_channels)
            )
            for edge_type in edge_types
        }

        # Define separate NNConv layers for each edge type
        self.convs = nn.ModuleList([
            HeteroConv({
                edge_type: NNConv(embedding_dim, out_channels, nn_dict[edge_type], aggr='mean')
                for edge_type in edge_types
            })
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # # Apply separate embeddings
        # x_dict = {node_type: self.embeddings[node_type](x)
        #           for node_type, x in x_dict.items()}

        x_dict = {node_type: self.node_type_linear[node_type](self.embeddings[node_type](x))
            for node_type, x in x_dict.items()}

        # Apply HeteroConv layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict

# model = BuySellLinkPrediction(num_nodes_dict, embedding_dim=64, num_edge_features=2, out_channels=32).to(device)
class BuySellLinkPrediction(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers):
        super(BuySellLinkPrediction, self).__init__()
        self.gnn = HeteroGNN(num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers=num_layers)
        self.linear = torch.nn.Linear(2 * embedding_dim + num_edge_features, 1)  # Linear layer for concatenated embeddings
        self.sigmoid = torch.nn.Sigmoid()

    # # Forward pass
    # preds = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict, edge_label_index, edge_label)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, edge_label_index, edge_label_attr):
        # Get embeddings from GNN
        out = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
        
        # Extract embeddings for 'congressperson' and 'ticker' node types
        congressperson_emb = out['congressperson']
        ticker_emb = out['ticker']
        
        # Concatenate embeddings of source and target nodes
        concatenated_emb = torch.cat([congressperson_emb[edge_label_index[0]], ticker_emb[edge_label_index[1]], edge_label_attr], dim=-1)
        
        # Compute predictions using linear layer and sigmoid activation
        preds = self.linear(concatenated_emb)
        preds = self.sigmoid(preds).squeeze()
        
        return preds
