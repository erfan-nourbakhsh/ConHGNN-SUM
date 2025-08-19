#!/usr/bin/env bash
# This script preprocesses a dataset for ConHGNN-SUM.
# It performs the following tasks:
# 1. Creates vocabulary from the training data.
# 2. Extracts low TF-IDF words from the training set.
# 3. Computes word-to-sentence (w2s) edge features for train, val, and test.
# 4. Optionally, computes word-to-document (w2d) edge features if task=multi.

dataset="$1"   # First argument: dataset name
datadir="$2"   # Second argument: path to dataset folder
task="$3"      # Third argument: task type (single/multi). Optional, default=single

set -e  # Exit immediately if a command exits with a non-zero status

# Check if dataset argument is provided
if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

# Check if data directory argument is provided
if [ ! -n "$datadir" ]; then
    echo "lose the data directory!"
    exit
fi

# Set default task to 'single' if not provided
if [ ! -n "$task" ]; then
    task=single
fi

type=(train val test)  # List of dataset splits

# Step 1: Create vocabulary from training data
echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
python script/createVoc.py --dataset $dataset --data_path $datadir/train.label.jsonl

# Step 2: Get low TF-IDF words from training data
echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
python script/lowTFIDFWords.py --dataset $dataset --data_path $datadir/train.label.jsonl

# Step 3: Compute word-to-sentence edge features for train, val, test
echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
for i in ${type[*]}
do
    python script/calw2sTFIDF.py --dataset $dataset --data_path $datadir/$i.label.jsonl
done

# Step 4: Optionally compute word-to-document edge features if task is multi
if [ "$task" == "multi" ]; then
    echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
    for i in ${type[*]}
    do
        python script/calw2dTFIDF.py --dataset $dataset --data_path $datadir/$i.label.jsonl
    done
fi

# Final message
echo -e "\033[34m[Shell] The preprocess of dataset $dataset has finished! \033[0m"
