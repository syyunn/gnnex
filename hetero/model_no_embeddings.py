import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import NNConv
from torch_geometric.nn import HeteroConv
from debug import NNConv

class HeteroGNNNoEmbedding(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers=2):
        super(HeteroGNNNoEmbedding, self).__init__()

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
                edge_type: NNConv(embedding_dim, out_channels, nn_dict[edge_type], aggr='max')
                for edge_type in edge_types
            })
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = {node_type: self.node_type_linear[node_type](x)
            for node_type, x in x_dict.items()}

        # Apply HeteroConv layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict

# model = BuySellLinkPrediction(num_nodes_dict, embedding_dim=64, num_edge_features=2, out_channels=32).to(device)
class BuySellLinkPredictionNoEmbedding(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers, num_pred_head_layers=2, hidden_dim=64):
        super(BuySellLinkPredictionNoEmbedding, self).__init__()
        self.gnn = HeteroGNNNoEmbedding(num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers=num_layers)

        # Build the prediction head layers
        prediction_head_layers = []
        prediction_head_layers.append(nn.Linear(2 * embedding_dim + num_edge_features, hidden_dim))
        prediction_head_layers.append(nn.ReLU())

        for _ in range(num_pred_head_layers - 1):
            prediction_head_layers.append(nn.Linear(hidden_dim, hidden_dim))
            prediction_head_layers.append(nn.ReLU())

        prediction_head_layers.append(nn.Linear(hidden_dim, 1))

        self.prediction_head = nn.Sequential(*prediction_head_layers)
        self.sigmoid = torch.nn.Sigmoid()

    # # Forward pass
    # preds = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict, edge_label_index, edge_label)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, edge_label_index, edge_label_attr):
        # print(x_dict)
        # print(edge_label_index.shape)
        # print(edge_label_attr.shape)
        # Get embeddings from GNN
        out = self.gnn(x_dict, edge_index_dict, edge_attr_dict)
        
        # Extract embeddings for 'congressperson' and 'ticker' node types
        congressperson_emb = out['congressperson']
        ticker_emb = out['ticker']
        
        # Concatenate embeddings of source and target nodes
        concatenated_emb = torch.cat([congressperson_emb[edge_label_index[0]], ticker_emb[edge_label_index[1]], edge_label_attr], dim=-1)
        
        # Compute predictions using linear layer and sigmoid activation
        preds = self.prediction_head(concatenated_emb)
        print("preds befoe sig", preds) 
        # print("preds befoe sig", preds)
        preds = self.sigmoid(preds).squeeze()
        
        return preds
