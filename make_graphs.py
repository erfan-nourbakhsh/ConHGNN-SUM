import os  # To run system commands

# Number of instances to process in each batch
offset = 1280

# Loop over instance indices from 12800 to 13000 in steps of `offset`
for i in range(12800, 13000, offset):
    # Run evaluation script with the current batch indices
    os.system(f"python evaluation.py --from_instances_index {i} --max_instances {offset}")
