import os  # For file and directory operations
from dgl.data.utils import load_graphs  # To load DGL graph objects from .bin files

# Walk through the directory containing test graph files
for root, _, files in os.walk("cache/CNNDM/graphs/test"):
    indexes = []  # List to store numerical indices of graph files

    # Iterate over all files in the current directory
    for file in files:
        # Extract the numeric index from filename (assuming filenames are like '1234.bin')
        from_index = int(file[:-4])
        indexes.append(from_index)  # Add index to the list

        path = os.path.join(root, file)  # Full path to the graph file

    # Find the maximum index among all files
    max_indexes = max(indexes)
    print(max_indexes, " ", max_indexes + 256)  # Display max index and max+batch size

    # Check for missing indices in steps of 256
    for i in range(0, max_indexes, 256):
        if i not in indexes:
            print("ERROR: ", i)  # Print any missing indices
