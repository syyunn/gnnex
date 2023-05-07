import pickle
from tqdm import tqdm

import torch
from torch_geometric.loader import LinkNeighborLoader

from torch.nn import functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from util import RandomLinkSplit
import torch_geometric.transforms as T
from util import RandomLinkSplitKfolds

# Define the model
from model import BuySellLinkPrediction

import csv

# Set the random seed for PyTorch, NumPy, and random
seed = 2328466898069313329
torch.manual_seed(seed)

# Print the random seed
print(f"Random seed: {torch.initial_seed()}")

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
print(data.node_types)

# Collect edge_types 
edge_types = []

# Convert edge_index tensors to integer type (torch.long)
for edge_type, edge_index in data.edge_index_dict.items():
    data.edge_index_dict[edge_type] = edge_index.to(torch.long)
    edge_types.append(edge_type)

print(edge_types)


from collections import defaultdict

# # define
# auc_rocs_train = defaultdict(dict)
# auc_rocs_test = defaultdict(dict)

# # define accs
# accs_train = defaultdict(dict)
# accs_test = defaultdict(dict)

# define subsets
import itertools

def generate_subsets(edge_types):
    subsets = [[]]  # Initialize with an empty subset
    for i in range(1, len(edge_types) + 1):
        subsets.extend(list(itertools.combinations(edge_types, i)))
    return subsets

edge_types_wo_revs = [edge_type for edge_type in edge_types if not edge_type[1].startswith("rev_")]

subsets = generate_subsets(edge_types_wo_revs)
print(len(subsets))
for subset in subsets:
    print(subset)

#read k_folds.pkl
import pickle
with open('k_folds.pkl', 'rb') as f:
    k_folds = pickle.load(f)

auc_rocs_train = k_folds[0]
accs_train = k_folds[1]

# define accs
auc_rocs_test = k_folds[2]
accs_test = k_folds[3]

already_done = k_folds[0].keys()
print(already_done)

for subset in tqdm(subsets):

    edge_type_include = subset
    key = tuple(edge_type_include) 
    if key in already_done:
        already_done_folds = list(k_folds[0][key].keys())
        if max(already_done_folds) == 4:
            continue
        else:
            pass

    print("Edge types to include: ", edge_type_include)

    edge_type_remove = [edge_type for edge_type in edge_types if edge_type not in edge_type_include]
    assert len(edge_type_include) + len(edge_type_remove) == len(edge_types)

    edge_type_include_rev = [(edge[2], 'rev_' + edge[1], edge[0]) for edge in edge_type_include]

    edge_type_include_w_rev = list(edge_type_include) + edge_type_include_rev


    model_edge_types = edge_type_include_w_rev

    print("Edge types:", edge_types)
    print("Total number of edge types: ", len(edge_types))

    # transform = T.RandomLinkSplit(
    # transform = RandomLinkSplit(
    transform = RandomLinkSplitKfolds(
        num_val=0,
        num_test=0.2,
        is_undirected=False, # with rev edges, the model recognizes the edge as "directed"
        # disjoint_train_ratio=0.3, # Across the training edges, we use 70% of edges for message passing, and 30% of edges for supervision.
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=("congressperson", "buy-sell", "ticker"),
        rev_edge_types=("ticker", "rev_buy-sell", "congressperson"),
    )


    # Create the CSV file with the specified column names, if it doesn't exist already
    csv_file_name = "log_results.csv"

    # write header if file does not exist
    # with open(csv_file_name, "a", newline="") as csvfile:
    #     fieldnames = ["manual_seed", "edge_type_removed", "fold", "accu", "auc_roc", "epoch", "train_test"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()

    fieldnames = ["manual_seed", "edge_type_removed", "fold", "accu", "auc_roc", "epoch", "train_test"]

    for fold in range(5):
        #train_data, val_data, test_data = transform(data)
        print("fold", fold)
        train_data, val_data, test_data = transform(data, fold=fold) # custom RandomLinkSplitKfolds

        #   Define seed edges:
        edge_label_index = train_data["congressperson", "buy-sell", "ticker"].edge_label_index
        edge_label = train_data["congressperson", "buy-sell", "ticker"].edge_label
        edge_attr = train_data["congressperson", "buy-sell", "ticker"].edge_attr

        # Create a dictionary to map edge indices to their attributes
        edge_to_attr = {(src.item(), dst.item()): attr.to(device) for src, dst, attr in zip(*edge_label_index, edge_attr)}

        # In the first hop, we sample at most 20 neighbors.
        # In the second hop, we sample at most 10 neighbors.

        num_neigbors = [20, 10, 5]
        # batch_size = 128
        # batch_size = 256
        batch_size = 4096
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

        # Given the HeteroData object named 'data'
        num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}

        # Print the num_nodes_dict
        print("num_nodes_dict:", num_nodes_dict)

        # Instantiate the model
        num_layers = 2
        print("num_layers", num_layers)
        # model = BuySellLinkPrediction(num_nodes_dict, embedding_dim=64, num_edge_features=2, out_channels=64, edge_types=edge_types, num_layers=num_layers).to(device)
        model = BuySellLinkPrediction(num_nodes_dict, embedding_dim=64, num_edge_features=2, out_channels=64, edge_types=model_edge_types, num_layers=num_layers).to(device)

        # Training loop
        epochs = 5
        # optimizer = optim.Adam(model.parameters(), lr=0.005)  # You can set the learning rate (lr) as needed
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can set the learning rate (lr) as needed

        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decay the learning rate by a factor of 0.1 every 10 epochs

        # Initialize a variable to keep track of the best test AUC-ROC score
        best_test_auc_roc = 0.0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_accuracy = 0
            total_auc_roc = 0

            for batch in tqdm(train_loader):
                batch = batch.to(device) # Move the batch to the device
                batch_edge_label = batch[("congressperson", "buy-sell", "ticker")].edge_label

                optimizer.zero_grad()
                
                # Get node ids from the batch
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
                preds, preds_before_sig = model(x_dict, batch.edge_index_dict, scaled_edge_attr_dict, edge_label_index, edge_label_attr=batch_edge_label_attr)
                # print("preds", preds)
                # Assuming 'preds' is a tensor obtained from the model's output
                num_ones = torch.sum(torch.eq(preds, 1)).item()
                num_zeros = torch.sum(torch.eq(preds, 0)).item()

                # Print the results
                # print(f"Number of ones in 'preds': {num_ones}")
                # print(f"Number of zeros in 'preds': {num_zeros}")
                
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

            # Print per-epoch loss
            avg_loss = total_loss / len(train_loader) # avg means avg over batches in an epoch
            avg_accuracy = total_accuracy / len(train_loader)
            avg_auc_roc = total_auc_roc / len(train_loader)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Train Accuracy: {avg_accuracy:.4f} Train AUC-ROC: {avg_auc_roc:.4f}")

            # with open(csv_file_name, "a", newline="") as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #     # Log the train metrics
            #     writer.writerow({
            #         "manual_seed": seed,
            #         "edge_type_removed": edge_type_removed,
            #         "fold": fold,
            #         "accu": avg_accuracy,
            #         "auc_roc": avg_auc_roc,
            #         "epoch": epoch,
            #         "train_test": "train",
            #     })

            auc_rocs_train[key][fold] = avg_auc_roc
            accs_train[key][fold] = avg_accuracy

            # eval
            # Evaluate the model on the test dataset
            from util import evaluate
            test_loss, test_accuracy, test_auc_roc = evaluate(test_loader, model, device, num_nodes_dict, test_data, test_edge_to_attr)
            print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f} Test AUC-ROC: {test_auc_roc:.4f}")

            # with open(csv_file_name, "a", newline="") as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #     # Log the test metrics
            #     writer.writerow({
            #         "manual_seed": seed,
            #         "edge_type_removed": edge_type_removed,
            #         "fold": fold,
            #         "accu": test_accuracy,
            #         "auc_roc": test_auc_roc,
            #         "epoch": epoch, # meaning logged after the finish of such epoch of train data
            #         "train_test": "test",
            #     })
            auc_rocs_test[key][fold] = test_auc_roc
            accs_test[key][fold] = test_accuracy

            # Check if the current test AUC-ROC score is better than the best one seen so far
            if test_auc_roc > best_test_auc_roc:
                # Update the best test AUC-ROC score
                best_test_auc_roc = test_auc_roc
                # Save the model with the best test AUC-ROC score
                print("Model saved with best test AUC-ROC:", best_test_auc_roc)
                torch.save(model.state_dict(), f"buysell_link_prediction_best_model_accu_{best_test_auc_roc}_new.pt")


            to_save = [auc_rocs_train, accs_train, auc_rocs_test, accs_test]
            # pickle to_save
            pickle_file_name = f"k_folds.pkl"
            with open(pickle_file_name, "wb") as f:
                pickle.dump(to_save, f)