import pandas as pd

# Read the CSV file
df = pd.read_csv('log_results.csv')

# Function to extract the second element of the first tuple in the list
def extract_second_element(edge_type_removed):
    if edge_type_removed.startswith("["):
        try:
            tuple_str = edge_type_removed[1:-1]
            first_tuple_str = tuple_str.split("),")[0] + ")"
            second_element = first_tuple_str.split(",")[1].strip()
            return second_element
        except IndexError:
            return edge_type_removed
    else:
        return edge_type_removed

# Apply the function to the 'edge_type_removed' column
df['edge_type_removed'] = df['edge_type_removed'].apply(extract_second_element)

# Write the modified DataFrame to a new CSV file
df.to_csv('modified_log_results.csv', index=False)
