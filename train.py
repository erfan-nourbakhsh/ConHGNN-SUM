import datetime  # For timestamps and logging
import os        # For file and path operations
import random    # For setting random seeds
import numpy as np  # For numeric operations and setting random seeds
import torch        # PyTorch for deep learning

from config import pars_args  # Function to parse command-line arguments / configs
from model_manager.model import Model  # The main model class (ConHGNN-SUM)
from data_manager import data_loaders  # For dataset loading and batching
from module.embedding import Word_Embedding  # To load pretrained embeddings
from module.vocabulary import Vocab  # Vocabulary object
from tools.logger import *  # Logger for info/debug/error messages
from runner.train import setup_training  # Function to run training loop
from utils import set_device  # Utility to set device (CPU/GPU)


def initial_seed(hps):
    """
    Initialize random seeds for reproducibility.

    Args:
        hps: Hyperparameter/config object containing `seed`.
    """
    random.seed(hps.seed)  # Seed Python's random module
    np.random.seed(hps.seed)  # Seed NumPy
    torch.manual_seed(hps.seed)  # Seed PyTorch CPU RNG


def get_files(hps):
    """
    Generate paths to dataset files, vocabulary, embeddings, and logs.

    Args:
        hps: Hyperparameter/config object.

    Returns:
        Tuple containing file paths for train, validation, vocab, filter words, 
        TF-IDF paths, logs, and graphs directory.
    """
    train_file = os.path.join(hps.data_dir, "train.label.jsonl")
    valid_file = os.path.join(hps.data_dir, "val.label.jsonl")
    vocal_file = os.path.join(hps.cache_dir, "vocab")
    filter_word = os.path.join(hps.cache_dir, "filter_word.txt")
    train_w2s_path = os.path.join(hps.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(hps.cache_dir, "val.w2s.tfidf.jsonl")
    graphs_path = os.path.join(hps.cache_dir, "graphs")
    log_path = hps.log_root

    return train_file, valid_file, vocal_file, filter_word, train_w2s_path, val_w2s_path, log_path, graphs_path


def main():
    """
    Main function for preparing data, embeddings, model, and training.
    """
    args = pars_args()  # Parse command-line arguments
    hps = args
    hps = set_device(hps=hps)  # Set device (CPU/GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = hps.gpu  # Specify GPU ID

    torch.set_printoptions(threshold=50000)  # Set tensor print threshold

    # Get all relevant file paths
    train_file, valid_file, vocal_file, filter_word, train_w2s_path, val_w2s_path, log_path, graphs_dir = get_files(hps=hps)

    # Setup logging directory
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # Current timestamp
    log_path = os.path.join(log_path, "train_" + now_time)
    file_handler = logging.FileHandler(log_path)  # File logging handler
    file_handler.setFormatter(formatter)  # Use predefined formatter
    logger.addHandler(file_handler)  # Add file handler to logger

    logger.info("Pytorch %s", torch.__version__)  # Log PyTorch version
    logger.info("[INFO] Create Vocab, vocab path is %s", vocal_file)
    vocab = Vocab(vocal_file, hps.vocab_size)  # Load vocabulary
    embed = torch.nn.Embedding(vocab.size(), hps.word_emb_dim, padding_idx=0)  # Initialize embedding layer

    # Load pretrained word embeddings if specified
    if hps.word_embedding:
        embed_loader = Word_Embedding(hps.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(hps.word_emb_dim)  # Load vectors
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, hps.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))  # Copy to embedding layer
        embed.weight.requires_grad = hps.embed_train  # Set whether embeddings are trainable

    logger.info(hps)  # Log hyperparameters

    # Handle HSG model
    if hps.model == "HSG":
        # Dictionary containing all necessary data variables
        data_variables = {
            "train_file": train_file,
            "valid_file": valid_file,
            "vocab": vocab,
            "filter_word": filter_word,
            "train_w2s_path": train_w2s_path,
            "val_w2s_path": val_w2s_path,
            "graphs_dir": graphs_dir
        }

        # Optional: fill graph cache for training
        if hps.fill_graph_cache:
            for i in range(1):
                with torch.no_grad():  # Disable gradient calculation for preprocessing
                    data_loaders.make_dataloader(
                        data_file=data_variables["train_file"],
                        vocab=data_variables["vocab"],
                        hps=hps,
                        filter_word=data_variables["filter_word"],
                        w2s_path=data_variables["train_w2s_path"],
                        graphs_dir=os.path.join(data_variables["graphs_dir"], "train")
                    )
                hps.from_instances_index += hps.max_instances
                print(f">>>>from:", hps.from_instances_index)

        else:
            # Initialize model
            model = Model(hps, embed)
            logger.info("[MODEL] ConHGNN-SUM ")

            # Start training
            setup_training(model=model, hps=hps, data_variables=data_variables)

    # Handle other model types (HDSG not implemented here)
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")


if __name__ == '__main__':
    main()  # Entry point
