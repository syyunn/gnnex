import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from explain import GNNExplainer_
# from torch_geometric.explain.algorithm import GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_networkx

# from model import Net
import numpy as np

# Load the dataset
dataset = KarateClub()
data = dataset[0]

# Set data.x to the one-hot encoded node labels
data.x = F.one_hot(data.y).to(torch.float)
print("data.x: ", data.x)

# get number of classes
num_classes = dataset.num_classes
print("num_classes: ", num_classes)

# Set data.y to None
data.train_mask = data.val_mask = data.test_mask = data.y = None

# Convert the graph to an undirected one
data.edge_index = to_undirected(data.edge_index)
print("data.edge_index:", data.edge_index)

# Create positive and negative edge indices
pos_edge_index = data.edge_index
neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)

# Define the model

# Add an inner product decoder to the existing model
print("dataset.num_features: ", dataset.num_features)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.x.size(1), 16) # 16 is node-representation size (hidden size)
        self.conv2 = GCNConv(16, 16)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # print(x.shape)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # print(x.shape)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        # print(pos_edge_index.shape)
        # print(neg_edge_index.shape)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # print("edge_index: ", edge_index.shape)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits


# Train GNN model
device = torch.device('cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

# Initialize the model and optimizer
device = torch.device('cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    logits = model.decode(z, pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


# Training loop
for epoch in range(1000):
    loss = train()
    print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")

explainer = GNNExplainer_(model=model, epochs=200) # simple init of the class.
edge_index_to_explain_src = 0
edge_index_to_explain_dst = 2
edge_index_to_explain = torch.tensor([[edge_index_to_explain_src], [edge_index_to_explain_dst]], dtype=torch.long).to(device)

# Find the corresponding edge in the undirected graph
edge_index_to_explain_undirected = None
for i in range(data.edge_index.size(1)):
    src, dst = data.edge_index[0][i], data.edge_index[1][i]
    if (src == edge_index_to_explain_src and dst == edge_index_to_explain_dst) or (src == edge_index_to_explain_dst and dst == edge_index_to_explain_src):
        # print("i: ", i)
        edge_index_to_explain_undirected = torch.tensor([[src], [dst]], dtype=torch.long).to(device)
        break
print("edge_index_to_explain_undirected: ", edge_index_to_explain_undirected)

node_mask, edge_mask = explainer.explain_edge(edge_index_to_explain_undirected, data.x, data.edge_index)
print("node mask:", node_mask)
print("edge mask:", edge_mask)

edge_src, edge_tgt = edge_index_to_explain_undirected[0][0].item(), edge_index_to_explain_undirected[1][0].item()
edge_index = data.edge_index
edge_mask_index = None
for i in range(edge_index.size(1)):
    src, tgt = edge_index[0][i], edge_index[1][i]
    if (src == edge_src and tgt == edge_tgt) or (src == edge_tgt and tgt == edge_src):
        edge_mask_index = i
        break

# Retrieve the corresponding edge mask value
if edge_mask_index is not None:
    edge_mask_value = edge_mask[edge_mask_index].item()
    print(f"The edge mask value for the edge ({edge_src}, {edge_tgt}) is: {edge_mask_value}")
else:
    print(f"The edge ({edge_src}, {edge_tgt}) was not found in the edge_index.")
    
# Create class labels dictionary from one-hot encoded labels
class_labels = {i: torch.argmax(data.x[i]).item() for i in range(data.x.size(0))}

# Define a function to get unique class labels
def unique_class_labels(class_labels):
    return set(class_labels.values())


# Define a function to get nodes belonging to a specific class
def nodes_of_class(class_labels, target_class):
    return [node for node, class_label in class_labels.items() if class_label == target_class]

# Define a dictionary to map class labels to node shapes
class_shape_map = {
    0: "o",  # Circle
    1: "s",  # Square
    2: "^",  # Triangle
    3: "D"   # Diamond
}

from matplotlib.colors import LinearSegmentedColormap

def visualize(node_mask, edge_mask):
    G = to_networkx(data, to_undirected=True)
    pos = nx.kamada_kawai_layout(G)

    node_map = {i: j for i, j in enumerate(list(G.nodes()))}

    # Create a dictionary to map edge pairs to their corresponding edge mask values
    edge_mask_dict = {(data.edge_index[0][i].item(), data.edge_index[1][i].item()): edge_mask[i].item() for i in range(data.edge_index.size(1))}

    # Use the edge_mask_dict to set the edge colors and widths
    edge_colors = [edge_mask_dict[edge] if edge in edge_mask_dict else 0 for edge in G.edges()]
    edge_widths = [10 if edge == (edge_index_to_explain_src, edge_index_to_explain_dst) or edge == (edge_index_to_explain_dst, edge_index_to_explain_src) else 1 for edge in G.edges()]

    # Normalize the node and edge mask values
    node_mask_normalized = (node_mask - node_mask.min() + 0.1) / (node_mask.max() - node_mask.min() + 0.1)
    # print("node_mask_normalized: ", node_mask_normalized)
    edge_colors_normalized = (np.array(edge_colors) - min(edge_colors)) / (max(edge_colors) - min(edge_colors))

    # Create custom color maps for nodes and edges
    cmap_nodes = plt.get_cmap('Reds')
    cmap_edges = plt.get_cmap('Reds')

    plt.figure(figsize=(16, 10))
    nx.draw_networkx_edges(G, pos, edge_color=cmap_edges(edge_colors_normalized), alpha=0.8, width=edge_widths)

    # Calculate node colors for all nodes
    all_node_colors = cmap_nodes(node_mask_normalized)

    # Find the closest index to node_mask_normalized[0] in node_mask_normalized (excluding the first element itself)
    first_val = node_mask_normalized[0]
    rest_vals = node_mask_normalized[1:]
    closest_index = torch.argmin(torch.abs(rest_vals - first_val)) + 1  # Add 1 to account for the excluded first element

    # Impute all_node_colors[0] with all_node_colors[MOST_CLOSE_VAL]
    all_node_colors[0] = all_node_colors[closest_index]

    # In the visualize function, replace the single nx.draw call with the following loop:
    for class_label in unique_class_labels(class_labels):
        nodes = nodes_of_class(class_labels, class_label)
        node_colors = all_node_colors[nodes] 
        # print("node_colors: ", node_colors)
        node_shape = class_shape_map[class_label]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=300, node_color=node_colors, node_shape=node_shape, alpha=0.8)

    nx.draw_networkx_labels(G, pos, labels=node_map, font_size=10)

    # Create legends for node and edge mask values
    sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=plt.Normalize(vmin=node_mask.min(), vmax=node_mask.max()))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes)
    cbar_nodes.ax.set_title("Node Mask", fontsize=12)

    sm_edges = plt.cm.ScalarMappable(cmap=cmap_edges, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges)
    cbar_edges.ax.set_title("Edge Mask", fontsize=12)

    plt.title("Explainer visualization", fontsize=14)
    plt.show()


# Visualize the graph with the node and edge masks
visualize(node_mask, edge_mask)
