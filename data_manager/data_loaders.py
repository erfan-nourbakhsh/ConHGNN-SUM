from torch.utils.data import BatchSampler

from data_manager.dataloader import SummarizationDataSet, graph_collate_fn
from data_manager.cached_dataset import CachedSummarizationDataSet
import torch


def make_dataloader(
    data_file,
    vocab,
    hps,
    filter_word,
    w2s_path,
    graphs_dir=None,
    from_index=0,
    to_index=None,
    shuffle=False
):
    """
    Creates a PyTorch DataLoader for summarization datasets, supporting both 
    cached graph datasets and raw text-based datasets.

    This function decides whether to load a `CachedSummarizationDataSet` (with preprocessed 
    graph data) or a `SummarizationDataSet` (with raw examples) based on `hps.use_cache_graph`. 
    It then wraps the dataset inside a PyTorch `DataLoader` for batching and iteration.

    Args:
        data_file (str): Path to the dataset JSON file containing examples.
        vocab (object): Vocabulary object for text tokenization and numericalization.
        hps (object): Hyperparameter settings, must include:
            - `use_cache_graph` (bool): Whether to use the cached graph dataset.
            - `batch_size` (int): Batch size for training/validation.
        filter_word (str): Path to a word filter file for preprocessing (used in `SummarizationDataSet`).
        w2s_path (str): Path to word-to-sentence mapping file.
        graphs_dir (str, optional): Path to directory containing cached graph files (.bin).
        from_index (int, optional): Starting index for slicing the dataset. Default is 0.
        to_index (int, optional): Ending index for slicing the dataset. Default is None (use full dataset).
        shuffle (bool, optional): Whether to shuffle dataset items. Default is False.

    Returns:
        torch.utils.data.DataLoader: A DataLoader object wrapping the chosen dataset.
    """

    # Choose dataset type depending on whether we use cached graphs
    if hps.use_cache_graph:
        # Use graph-cached dataset (faster, preprocessed)
        dataset = CachedSummarizationDataSet(
            hps=hps,
            graphs_dir=graphs_dir,
            vocab=vocab,
            data_path=data_file,
            from_index=from_index,
            to_index=to_index
        )
    else:
        # Use raw summarization dataset (processes data on the fly)
        dataset = SummarizationDataSet(
            data_path=data_file,
            vocab=vocab,
            filter_word_path=filter_word,
            w2s_path=w2s_path,
            hps=hps,
            graphs_dir=graphs_dir
        )

    # Wrap dataset in a PyTorch DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hps.batch_size,   # Batch size defined in hyperparameters
        shuffle=shuffle,             # Shuffle only if specified
        num_workers=0,               # Single-threaded (can be increased for speed)
        collate_fn=graph_collate_fn  # Custom function to collate graphs and examples into a batch
    )

    # Explicitly delete dataset reference (free memory, DataLoader keeps its own handle)
    del dataset

    # Return DataLoader for training/validation/testing loops
    return loader
