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
                edge_type: NNConv(embedding_dim, out_channels, nn_dict[edge_type], aggr='max')
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
        # print("x_dict", x_dict)

        # Apply HeteroConv layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict

# model = BuySellLinkPrediction(num_nodes_dict, embedding_dim=64, num_edge_features=2, out_channels=32).to(device)
class BuySellLinkPrediction(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers, num_pred_head_layers=2, hidden_dim=64):
        super(BuySellLinkPrediction, self).__init__()
        self.gnn = HeteroGNN(num_nodes_dict, embedding_dim, num_edge_features, out_channels, edge_types, num_layers=num_layers)

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
        
        missing = []
        # Extract embeddings for 'congressperson' and 'ticker' node types
        try:
            congressperson_emb = out['congressperson']
        except KeyError:
            congressperson_emb = self.gnn.embeddings['congressperson']
            missing.append('congressperson')

        try:
            ticker_emb = out['ticker']
        except KeyError:
            ticker_emb = self.gnn.embeddings['ticker'] # these try-except for ablation studies where we remove certain types of edges.
            missing.append('ticker')
        
        # Concatenate embeddings of source and target nodes
        try:
            concatenated_emb = torch.cat([congressperson_emb[edge_label_index[0]], ticker_emb[edge_label_index[1]], edge_label_attr], dim=-1)
        except TypeError:
            if 'congressperson' in missing and 'ticker' in missing:
                concatenated_emb = torch.cat([congressperson_emb.forward(edge_label_index[0]), ticker_emb.forward(edge_label_index[1]), edge_label_attr], dim=-1)
            elif 'congressperson' in missing:
                concatenated_emb = torch.cat([congressperson_emb.forward(edge_label_index[0]), ticker_emb[edge_label_index[1]], edge_label_attr], dim=-1)
            elif 'ticker' in missing:
                concatenated_emb = torch.cat([congressperson_emb[edge_label_index[0]], ticker_emb.forward(edge_label_index[1]), edge_label_attr], dim=-1)
        
        # Compute predictions using linear layer and sigmoid activation
        preds_before_sig = self.prediction_head(concatenated_emb)
        # print("preds befoe sig", preds)
        preds = self.sigmoid(preds_before_sig).squeeze()
        
        return preds, preds_before_sig
