import os
import torch.utils.data
from dgl.data.utils import load_graphs

from data_manager.dataloader import Example, read_json


class CachedSummarizationDataSet(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling cached summarization data along with 
    preprocessed graph structures (Heterogeneous Summary Graphs, HSG).

    This dataset supports:
    - Loading textual examples and summaries from JSON files.
    - Caching and retrieving graph data in DGL binary format (.bin).
    - Providing access to examples and their corresponding graphs in a PyTorch Dataset format.

    Attributes:
        hps (object): Hyperparameter settings with attributes like sent_max_len, doc_max_timesteps, etc.
        sent_max_len (int): Maximum sentence length allowed.
        doc_max_timesteps (int): Maximum number of timesteps for documents.
        max_instance (int): Maximum number of instances to load.
        graphs_dir (str): Path to the directory where graph files (.bin) are stored.
        use_cache (bool): Whether to use cached graphs or not.
        from_index (int): Starting index of the dataset slice.
        to_index (int): Ending index of the dataset slice.
        graph_index_from (int): Current starting index of loaded graphs.
        graph_index_offset (int): Number of graphs loaded at once (batching offset).
        size (int): Total dataset size after applying limits.
        graphs (dict): Dictionary mapping indices to loaded DGL graphs.
        example_list (list): Cached list of textual examples.
        vocab (object): Vocabulary object for tokenization.
        data_path (str): Path to the dataset JSON file.
    """

    def __init__(self, hps, data_path=None, vocab=None, graphs_dir=None, from_index=0, to_index=0):
        # Store hyperparameters
        self.hps = hps
        self.sent_max_len = hps.sent_max_len  # Maximum length of a sentence
        self.doc_max_timesteps = hps.doc_max_timesteps  # Maximum timesteps in a document
        self.max_instance = hps.max_instances  # Maximum number of dataset instances
        self.graphs_dir = graphs_dir  # Directory storing graphs
        self.use_cache = self.hps.fill_graph_cache  # Whether to cache graphs
        self.from_index = from_index  # Dataset slice start index
        self.to_index = to_index  # Dataset slice end index
        self.graph_index_from = 0  # Current starting graph index
        self.graph_index_offset = 256  # Number of graphs loaded per chunk

        # Get list of files inside the graph directory
        root, _, files = list(os.walk(self.graphs_dir))[0]

        # Extract numeric indexes from filenames (assumes filenames like "123.bin")
        indexes = [int(item[:-4]) for item in files]
        max_index = max(indexes)  # Highest index among available graph files

        # Load the graphs at max index to compute dataset size
        g, label_dict = load_graphs(os.path.join(root, f"{max_index}.bin"))
        size = max_index + len(g) - 1  # Total size of graph dataset

        # Set dataset slice boundary
        if to_index is None:
            to_index = size

        # Ensure the dataset respects maximum instance limits
        max_instances = hps.max_instances if hps.max_instances else 288000
        self.size = min([to_index - from_index, max_instances, size])

        # Initialize storage for graphs
        self.graphs = dict()
        self.load_HSG_graphs()  # Preload initial graphs

        # Example list (lazy loaded later)
        self.example_list = None
        self.vocab = vocab
        self.data_path = data_path

    def fill_example_list(self):
        """
        Loads examples from the dataset JSON file into memory.

        Reads from `data_path` and fills `self.example_list` with preprocessed text examples.
        """
        self.example_list = read_json(
            self.data_path,
            max_instance=self.max_instance,
            from_instances_index=self.hps.from_instances_index
        )

    def get_example(self, index):
        """
        Retrieve a single example by index.

        Args:
            index (int): Index of the example.

        Returns:
            Example: Processed example containing text, summary, and labels.
        """
        # If examples not loaded yet, load them
        if self.example_list is None:
            self.fill_example_list()

        # Retrieve the example dictionary
        e = self.example_list[index]

        # Ensure 'summary' field exists
        e["summary"] = e.setdefault("summary", [])

        # Wrap into Example object
        example = Example(
            e["text"], 
            e["summary"], 
            self.vocab, 
            self.sent_max_len, 
            e["label"]
        )
        return example

    def load_HSG_graphs(self):
        """
        Loads a batch of graphs from file into memory, starting at `graph_index_from`.

        Graphs are loaded in chunks determined by `graph_index_offset`.
        """
        # Load graphs from file
        graphs, _ = load_graphs(os.path.join(self.graphs_dir, f"{self.graph_index_from}.bin"))

        # Store graphs in dictionary with index-based keys
        for i, graph in enumerate(graphs):
            self.graphs[self.graph_index_from + i] = graph

    def get_graph(self, index):
        """
        Retrieve a graph by index, loading it from disk if not already cached.

        Args:
            index (int): Graph index.

        Returns:
            DGLGraph: The requested graph.
        """
        # If graph not cached, load the corresponding batch
        if index not in self.graphs.keys():
            self.graph_index_from = (index // self.graph_index_offset) * self.graph_index_offset
            self.load_HSG_graphs()

        return self.graphs[index]

    def __getitem__(self, index):
        """
        Retrieve dataset item by index (graph + index).

        Args:
            index (int): Index of dataset item.

        Returns:
            tuple: (graph, index) if successful, else None.
        """
        try:
            G = self.get_graph(index)  # Fetch the graph
            return G, index
        except Exception as e:
            print(f"EXCEPTION => {e}")
            return None

    def __getitems__(self, possibly_batched_index):
        """
        Retrieve multiple dataset items by a list of indices.

        Args:
            possibly_batched_index (list[int]): List of indices.

        Returns:
            list: List of (graph, index) pairs.
        """
        result = []
        for index in possibly_batched_index:
            item = self.__getitem__(self.from_index + index)
            if item is not None:
                result.append(item)

        return result

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: Dataset size.
        """
        return self.size
