import torch
from model import BuySellLinkPrediction
from util import get_edge_attr_for_batch
import pickle

# Open the file in binary read mode and unpickle the data
with open('./data/hetero_graph_data.pkl', "rb") as f:
    loaded_data = pickle.load(f)

# Extract the data from the loaded dictionary
data = loaded_data["hetero_graph"]
unique_tickers = loaded_data["unique_tickers"]
unique_congresspeople = loaded_data["unique_congresspeople"]
unique_committees = loaded_data["unique_committees"]
unique_bills = loaded_data["unique_bills"]
unique_naics = loaded_data["unique_naics"]

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

# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))

# Given the HeteroData object named 'data'
num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}

# Print the num_nodes_dict
print(num_nodes_dict)

# Instantiate the model
num_layers = 2

# Collect edge_types 
edge_types = []
# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    edge_types.append(edge_type)

print("Edge types:", edge_types)
print(len(edge_types))

model = BuySellLinkPrediction(num_nodes_dict, 
                              embedding_dim=64, 
                              num_edge_features=2, 
                              out_channels=64, 
                              edge_types=edge_types, 
                              num_layers=num_layers).to(device)

# Print the model architecture
print(model)

# Load the trained model
model_path = "buysell_link_prediction_best_model.pt"
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

which_edge = 0
congressperson_id, ticker_id = data[('congressperson', 'buy-sell', 'ticker')]['edge_index'][:, which_edge]
edge_label_index = torch.stack([congressperson_id, ticker_id], dim=0)

# Create a dictionary to map edge pairs to their attributes
edge_to_attr = {}
for key, edge_index in data.edge_index_dict.items():
    edge_attr = data.edge_attr_dict[key]
    for i, (src, dst) in enumerate(edge_index.t()):
        edge_to_attr[(src.item(), dst.item())] = edge_attr[i]

# date scaling
from datetime import date

start_date = date(2016, 1, 1)
today = date.today()
total_days = (today - start_date).days

# Create edge_label_attr tensor
edge_attr = torch.tensor(edge_to_attr[(congressperson_id.item(), ticker_id.item())]/total_days, dtype=torch.float, device=device)
print("Edge label attribute:", edge_attr)
# Prepare the input data for the model
x_dict = {node_type: data[node_type].node_id for node_type in num_nodes_dict.keys()}


scaled_edge_attr_dict = {key: value / total_days for key, value in data.edge_attr_dict.items()}

# Perform inference using the trained model
with torch.no_grad():
    # preds = model(x_dict, batch.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=batch_edge_label_attr)
    preds = model(x_dict, data.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=edge_attr)

# Print the prediction results
print("Prediction result:", preds.item())
