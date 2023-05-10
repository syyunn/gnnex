import pickle
from datetime import datetime, timedelta
import torch

# Define the reference date
ref_date = datetime.strptime("2016-01-01", "%Y-%m-%d")

upper_limit_date = datetime.now()

# Load the data object from the pickle file
with open('./data/hetero_graph_data.pkl', "rb") as f:
    loaded_data = pickle.load(f)

# Access the 'hetero_graph' key, which may contain the HeteroData object
hetero_graph = loaded_data['hetero_graph']

    # Iterate over each edge type and calculate the number of edges
for edge_type in [('congressperson', 'buy-sell', 'ticker'), ('congressperson', 'assignment', 'committee'), ('ticker', 'lobbies_on', 'bill'), ('bill', 'assigned_to', 'committee'), ('ticker', 'classified', 'naics'), ('ticker', 'rev_buy-sell', 'congressperson'), ('committee', 'rev_assignment', 'congressperson'), ('bill', 'rev_lobbies_on', 'ticker'), ('committee', 'rev_assigned_to', 'bill'), ('naics', 'rev_classified', 'ticker')]:
    # Get the edge index tensor for the current edge type
    edge_index = hetero_graph[edge_type]['edge_index']
    
    # Calculate the number of edges (length of edge index tensor along the second dimension)
    num_edges = edge_index.size(1)
    
    # Print the number of edges for the current edge type
    print(f"Number of edges for edge type '{edge_type}': {num_edges}")

    edge_attr = hetero_graph[edge_type]['edge_attr']
    elapsed_days = edge_attr[:, 0]
    
    # Convert elapsed days to actual dates using the reference date
    actual_dates = [ref_date + timedelta(days=int(d)) for d in elapsed_days]
    
    # Filter out dates that exceed the upper limit
    filtered_dates = [d for d in actual_dates if d <= upper_limit_date]
    
    # Calculate the minimum and maximum dates from the filtered dates
    earliest_date = min(filtered_dates)
    latest_date = max(filtered_dates)
    
    # Print the earliest and latest dates for the current edge type
    print(f"Earliest date for edge type '{edge_type}': {earliest_date.strftime('%Y-%m-%d')}")
    print(f"Latest date for edge type '{edge_type}': {latest_date.strftime('%Y-%m-%d')}")
