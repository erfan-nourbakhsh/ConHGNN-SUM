import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from dgl.data.utils import save_graphs, load_graphs
from nltk.corpus import stopwords
import time
import json
from collections import Counter
import numpy as np
import torch
import torch.utils.data
import dgl
from tools.logger import *

# Define filter words: NLTK stopwords + common punctuations
FILTER_WORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'',
                '`', '``', '-', '--', '|', '\/']
FILTER_WORD.extend(punctuations)


class Example(object):
    """
    Represents a single-document extractive summarization example.
    Handles sentence tokenization, word-to-id conversion, padding, and label matrix creation.
    """

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """
        Initialize Example object.

        Args:
            article_sents (list[str] or list[list[str]]): Sentences of article(s). Multi-document if list of lists.
            abstract_sents (list[str]): Sentences of summary/abstract.
            vocab (object): Vocabulary object, supports word2id mapping.
            sent_max_len (int): Maximum number of tokens per sentence.
            label (list[int]): Indices of selected sentences (e.g., [1,3,5]).
        """
        self.sent_max_len = sent_max_len
        self.enc_sent_len = []  # stores length of each sentence before padding
        self.enc_sent_input = []  # stores tokenized word ids
        self.enc_sent_input_pad = []  # stores padded word ids for each sentence

        # Store original sentences and abstract
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Flatten multi-document input into single list
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)

        # Convert words to ids and store lengths
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])

        # Pad sentences to maximum length
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Create label matrix
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1

    def _pad_encoder_input(self, pad_id):
        """
        Pad each sentence to maximum sentence length using pad_id.

        Args:
            pad_id (int): Vocabulary ID for padding token.
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class Example2(Example):
    """
    Multi-document summarization example, inherits from Example.
    Handles multiple documents and concatenates sentence embeddings per document.
    """

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)

        cur = 0
        self.original_articles = []
        self.article_len = []  # sentence count per document
        self.enc_doc_input = []  # concatenated word ids per document

        # Process each document separately
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


class SummarizationDataSet(torch.utils.data.Dataset):
    """
    PyTorch dataset for single-document extractive summarization.
    Supports filtering words, TFIDF mapping, and DGL graph creation for each example.
    """

    def __init__(self, data_path, vocab, filter_word_path, w2s_path, hps, graphs_dir=None):
        """
        Load examples, filter words, and optional cached graphs.

        Args:
            data_path (str): JSON dataset file path.
            vocab (object): Vocabulary object.
            filter_word_path (str): Path to custom filter words file.
            w2s_path (str): Path to word-to-sentence TFIDF JSON.
            hps (object): Hyperparameters object.
            graphs_dir (str, optional): Directory to save/load graphs.
        """
        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = hps.sent_max_len
        self.doc_max_timesteps = hps.doc_max_timesteps
        self.max_instance = hps.max_instances

        # Load JSON dataset
        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = read_json(data_path, max_instance=self.max_instance,
                                      from_instances_index=hps.from_instances_index)
        logger.info("[INFO] Finish reading %s. Total time: %.2f, size: %d",
                    self.__class__.__name__, time.time() - start, len(self.example_list))
        self.size = len(self.example_list)

        # Load filter words and map to vocabulary ids
        logger.info("[INFO] reading filter word File %s", filter_word_path)
        tfidf_w = read_text(filter_word_path)
        self.filter_words = FILTER_WORD
        self.filter_ids = [vocab.word2id(w.lower()) for w in FILTER_WORD]
        self.filter_ids.append(vocab.word2id("[PAD]"))  # keep "[UNK]" but remove "[PAD]"
        filter_words = list(set(tfidf_w).intersection(set(vocab._word_to_id.keys())))[:5000]
        self.filter_words += filter_words
        self.filter_ids += [self.vocab._word_to_id[word] for word in filter_words]

        # Load word-to-sentence TFIDF
        logger.info("[INFO] Loading word2sent TFIDF file from %s!", w2s_path)
        self.w2s_tfidf = read_json(w2s_path, max_instance=self.max_instance,
                                   from_instances_index=hps.from_instances_index)

        self.graphs_dir = graphs_dir
        self.use_cache = self.hps.use_cache_graph
        self.fill_cache = self.hps.fill_graph_cache
        if self.fill_cache:
            self.cache_graphs()

    def get_example(self, index):
        """
        Get an Example object for a given index.

        Args:
            index (int): Index of example.

        Returns:
            Example object
        """
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def pad_label_m(self, label_matrix):
        """
        Pad label matrix to maximum document timesteps.

        Args:
            label_matrix (np.array): Original label matrix.

        Returns:
            np.array: Padded label matrix
        """
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def add_word_node(self, graph, input_id):
        """
        Add word nodes to DGL graph, ignoring filtered words.

        Args:
            graph (dgl.DGLGraph): Graph object.
            input_id (list[list[int]]): Word ids per sentence.

        Returns:
            tuple: wid2nid, nid2wid mapping
        """
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in input_id:
            for wid in sentid:
                if wid not in self.filter_ids and wid not in wid2nid:
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        graph.add_nodes(w_nodes)
        graph.set_n_initializer(dgl.init.zero_initializer)
        graph.ndata["unit"] = torch.zeros(w_nodes)
        graph.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        graph.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def create_graph(self, input_pad, label, w2s_w):
        """
        Create a DGL graph for a single-document example.

        Args:
            input_pad (list[list[int]]): Padded word ids per sentence.
            label (np.array): Padded label matrix.
            w2s_w (dict): Word-to-sentence TFIDF mapping.

        Returns:
            dgl.DGLGraph: Constructed graph.
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.add_word_node(G, input_pad)
        w_nodes = len(nid2wid)

        # Add sentence nodes
        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        G.set_e_initializer(dgl.init.zero_initializer)

        # Add edges between word and sentence nodes
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw:
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

            # Add optional sentence-to-sentence edges
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)

        return G

    def get_graph(self, index):
        """
        Return a graph for a single example.

        Args:
            index (int): Index of example.

        Returns:
            dgl.DGLGraph
        """
        item = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]
        G = self.create_graph(input_pad, label, w2s_w)
        return G

    def __getitem__(self, index):
        """
        PyTorch Dataset __getitem__.

        Returns:
            tuple: (graph, index)
        """
        G = self.get_graph(index)
        return G, index

    def __getitems__(self, possibly_batched_index):
        """
        Load multiple graphs in parallel using ProcessPoolExecutor.
        """
        data = []
        with ProcessPoolExecutor(max_workers=1) as executor:
            for index, item in enumerate(executor.map(self.__getitem__, possibly_batched_index)):
                data.append(item)
        return data

    def _make_graphs_and_save(self, index):
        """
        Make graphs for a batch of examples and save to disk.
        """
        print(f"make graph for index = {index}")
        t1 = time.time()
        graphs = [item[0] for item in
                  self.__getitems__(list(range(index, min(index + 256, len(self.example_list)))))]
        path = os.path.join(self.graphs_dir, f"{index + self.hps.from_instances_index}.bin")
        save_graphs(path, graphs)
        del graphs
        print(f"save graphs {index} to {index + 255} =>  {time.time() - t1}")

    def cache_graphs(self):
        """
        Cache all example graphs to disk in parallel processes.
        """
        processes = []
        for index in range(0, len(self.example_list), 256):
            p = multiprocessing.Process(target=self._make_graphs_and_save, args=(index,))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
            process.close()
            del process
        del processes
        logger.info("finish save graphs")

    def __len__(self):
        return self.size


# -------------------------
# Helper functions
# -------------------------

def catDoc(textlist):
    """
    Concatenate lists of word ids from multiple sentences/documents into a single list.
    """
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def read_json(fname, max_instance=None, from_instances_index=None):
    """
    Read JSON lines file and return a list of dicts.

    Args:
        fname (str): File path.
        max_instance (int): Maximum number of instances to read.
        from_instances_index (int): Index to start reading from.

    Returns:
        list[dict]
    """
    data_list = []
    count = 0
    if from_instances_index is None:
        from_instances_index = 0
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
        for data in lines[from_instances_index:]:
            if max_instance is not None and count >= max_instance:
                return data_list
            data_list.append(json.loads(data))
            count += 1
    return data_list


def read_text(file_name):
    """
    Read a plain text file line by line.

    Returns:
        list[str]
    """
    data = []
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def graph_collate_fn(samples):
    """
    Custom collate function for batching DGL graphs.
    Sorts by number of sentence nodes descending.

    Args:
        samples (list[tuple]): List of (graph, index) pairs.

    Returns:
        tuple: batched_graph, sorted_indices
    """
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
