import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

from tqdm import tqdm

# Evaluation function
def evaluate(loader, model, device, num_nodes_dict):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)  # Move the batch to the device

            # Get node embeddings from the batch
            x_dict = {node_type: batch[node_type].node_id for node_type in num_nodes_dict.keys()}

            # Get edge_label_index and edge_label
            edge_label_index = batch[("congressperson", "buy-sell", "ticker")].edge_label_index
            edge_label = batch[("congressperson", "buy-sell", "ticker")].edge_label

            # Forward pass
            preds = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict, edge_label_index)

            # Compute loss
            loss = F.binary_cross_entropy(preds, edge_label.float())
            total_loss += loss.item()

            # Convert predicted probabilities to binary predictions
            binary_preds = (preds > 0.5).float()

            # Compute accuracy
            accuracy = accuracy_score(edge_label.cpu().numpy(), binary_preds.cpu().numpy())
            total_accuracy += accuracy

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    return avg_loss, avg_accuracy