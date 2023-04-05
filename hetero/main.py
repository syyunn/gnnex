import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.utils import negative_sampling

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Assign arbitrary node types (0 and 1)
node_types = torch.randint(0, 2, (data.num_nodes,))

# Assign arbitrary edge types (0 and 1)
edge_types = torch.randint(0, 2, (data.edge_index.size(1),))

data.node_types = node_types
data.edge_types = edge_types

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(dataset.num_features, 64, num_relations=2, num_bases=30)
        self.conv2 = RGCNConv(64, 64, num_relations=2, num_bases=30)
        self.gat = GATConv(64, 1, heads=1)

    def forward(self, x, edge_index, edge_types):
        x = F.relu(self.conv1(x, edge_index, edge_types))
        x = self.conv2(x, edge_index, edge_types)
        return self.gat(x, edge_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()

    # Get positive links for training
    pos_train_edge_index = data.edge_index

    # Sample negative links for training
    num_neg_samples = pos_train_edge_index.size(1)
    neg_train_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes, num_neg_samples=num_neg_samples, method="sparse")

    pos_pred = model(data.x, pos_train_edge_index, data.edge_types)
    neg_pred = model(data.x, neg_train_edge_index, data.edge_types)

    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones(pos_pred.size()).to(device))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros(neg_pred.size()).to(device))

    loss = (pos_loss + neg_loss) / 2
    loss.backward()
    print(loss)
    optimizer.step()

# Train the model
for epoch in range(100):
    train()
