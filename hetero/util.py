import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm
import random

def get_edge_attr_for_batch(data_edge_index, data_edge_attr, batch_edge_label_index, edge_to_attr):
    batch_size = batch_edge_label_index.shape[1]
        
    # Initialize the tensor to store batch_edge_attr
    batch_edge_attr = torch.zeros((batch_size, data_edge_attr.shape[1]))
    
    # Iterate through the batch_edge_label_index and retrieve the corresponding edge attributes
    for i, (src, dst) in enumerate(batch_edge_label_index.t()):
        try:
            batch_edge_attr[i] = edge_to_attr[(src.item(), dst.item())]
        except KeyError:
            batch_edge_attr[i] = data_edge_attr[random.choice(range(data_edge_attr.shape[0]))]
    
    return batch_edge_attr



# Evaluation function
def evaluate(loader, model, device, num_nodes_dict, test_data, edge_to_attr):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_auc_roc = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)  # Move the batch to the device

            # Get node embeddings from the batch
            x_dict = {node_type: batch[node_type].node_id for node_type in num_nodes_dict.keys()}

            # Get edge_label_index and edge_label
            edge_label_index = batch[("congressperson", "buy-sell", "ticker")].edge_label_index
            edge_label = batch[("congressperson", "buy-sell", "ticker")].edge_label
            # print("edge_label", edge_label)

            # Count the number of samples with label 0 (negative samples)
            num_negatives = torch.sum(edge_label == 0).item()

            # Count the number of samples with label 1 (positive samples)
            num_positives = torch.sum(edge_label == 1).item()

            # Print the counts
            print("This is Test session")
            print(f"Number of negative samples (label 0): {num_negatives}")
            print(f"Number of positive samples (label 1): {num_positives}")

            from util import get_edge_attr_for_batch
            batch_edge_label_attr = get_edge_attr_for_batch(test_data["congressperson", "buy-sell", "ticker"].edge_index, test_data["congressperson", "buy-sell", "ticker"].edge_attr, edge_label_index, edge_to_attr)
            batch_edge_label_attr = batch_edge_label_attr.to(device)

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

            # Compute loss
            loss = F.binary_cross_entropy(preds, edge_label.float())
            total_loss += loss.item()

            # Convert predicted probabilities to binary predictions
            binary_preds = (preds > 0.5).float()

            # Compute accuracy
            accuracy = accuracy_score(edge_label.cpu().numpy(), binary_preds.cpu().numpy())
            total_accuracy += accuracy

            # Compute AUC-ROC if there are both positive and negative samples
            import numpy as np
            unique_labels = np.unique(edge_label.cpu().numpy())
            if len(unique_labels) > 1:
                auc_roc = roc_auc_score(edge_label.cpu().numpy(), preds.squeeze().cpu().detach().numpy())
                total_auc_roc += auc_roc
                num_batches += 1

            # # Compute AUC-ROC
            # auc_roc = roc_auc_score(edge_label.cpu().numpy(), preds.squeeze().cpu().detach().numpy())
            # total_auc_roc += auc_roc


    # Calculate average loss and accuracy
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    # avg_auc_roc = total_auc_roc / len(loader)
    avg_auc_roc = total_auc_roc / num_batches if num_batches > 0 else None

    return avg_loss, avg_accuracy, avg_auc_roc
