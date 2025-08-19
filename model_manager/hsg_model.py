import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn  # utilities for handling variable-length sequences in RNNs
import dgl  # Deep Graph Library for graph neural networks

# Custom modules
from module.Encoder import sentEncoder  # CNN/LSTM-based sentence encoder
from module.GAT import WSWGAT  # Word-to-sentence and sentence-to-word GAT module
from module.PositionEmbedding import get_sinusoid_encoding_table  # positional encoding for sentences
from dgl.data.utils import save_graphs  # to save DGL graphs
from dgl.nn.pytorch import GATConv  # Graph Attention Network layer


class HSumGraph(nn.Module):
    """
    Hierarchical Summarization Graph model.
    This model builds a hierarchical word-sentence graph for extractive summarization.
    - Word nodes and sentence nodes are linked via GAT-based attention.
    - Iterative word-sentence and sentence-word updates are applied.
    - Residual connections are added; sent2sent edges are omitted.
    """
    def __init__(self, hps, embed):
        """
        Initializes HSumGraph.

        :param hps: hyperparameters object
        :param embed: torch.nn.Embedding, word embedding matrix
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter  # number of graph update iterations
        self._embed = embed.to(self._hps.device)  # move embedding to device
        self.embed_size = hps.word_emb_dim  # word embedding dimension

        # Initialize sentence node-related parameters
        self._init_sn_param()

        # Embedding for TF-IDF box values
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)  # 10 possible tfidf bins

        # Projection layer for concatenated sentence features
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # Word-to-sentence graph attention
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(
            in_dim=embed_size,
            out_dim=hps.hidden_size,
            num_heads=hps.n_head,
            attn_drop_out=hps.atten_dropout_prob,
            ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
            ffn_drop_out=hps.ffn_dropout_prob,
            feat_embed_size=hps.feat_embed_size,
            layerType="W2S"
        )

        # Sentence-to-word graph attention
        self.sent2word = WSWGAT(
            in_dim=hps.hidden_size,
            out_dim=embed_size,
            num_heads=6,
            attn_drop_out=hps.atten_dropout_prob,
            ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
            ffn_drop_out=hps.ffn_dropout_prob,
            feat_embed_size=hps.feat_embed_size,
            layerType="S2W"
        )

        # Number of features for node classification (unused here)
        self.n_feature = hps.hidden_size

        # Move entire model to device
        self.to(hps.device)

    def forward(self, graph):
        """
        Forward pass for hierarchical summarization.

        :param graph: DGLGraph, batched graph with word and sentence nodes
            node features:
                - word: unit=0, dtype=0, id=word index
                - sentence: unit=1, dtype=1, words tensor, position, label tensor
            edges:
                - word2sent, sent2word: tfidf embedding
        :return: sentence states tensor [num_sentence_nodes, hidden_size]
        """

        # Initialize word node features
        word_feature = self.set_wnfeature(graph)  # [num_word_nodes, embed_size]

        # Initialize sentence node features and project to hidden_size
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [num_sentence_nodes, hidden_size]

        # Initial states for iterative updates
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        # Iterative updates between word and sentence nodes
        for i in range(self._n_iter):
            # Update word states based on sentence states
            word_state = self.sent2word(graph, word_state, sent_state)
            # Update sentence states based on word states
            sent_state = self.word2sent(graph, word_state, sent_state)

        # Return final sentence representations
        return sent_state

    def _init_sn_param(self):
        """
        Initialize sentence node parameters:
        - positional embeddings
        - CNN/LSTM sentence encoders
        - projection layers
        """
        # Sinusoidal positional embeddings for sentences
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True
        )

        # Linear projection for CNN features
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)

        # LSTM encoder for sentences
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(
            self.embed_size,
            self.lstm_hidden_state,
            num_layers=self._hps.lstm_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=self._hps.bidirectional
        )

        # Linear projection after LSTM (consider bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        # N-gram CNN sentence encoder
        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        """
        Compute CNN-based sentence feature with positional embeddings.

        :param graph: DGLGraph
        :param snode_id: sentence node IDs
        :return: cnn_feature tensor [num_sentence_nodes, n_feature_size]
        """
        # Extract CNN features using ngram encoder
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])
        # Store sentence embeddings in graph
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        # Add positional embeddings
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)
        position_embedding = self.sent_pos_embed(snode_pos)
        # Linear projection of combined features
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        """
        Compute LSTM-based sentence features for batched sentences.

        :param features: list of sentence embeddings per graph
        :param glen: list of sentence counts per graph
        :return: lstm_feature tensor [total_sentence_nodes, n_feature_size]
        """
        # Pad sequences to equal length
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        # Pack sequences for LSTM
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_input)
        # Unpack sequences
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # Concatenate outputs for each sentence
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))
        return lstm_feature

    def set_wnfeature(self, graph):
        """
        Initialize word node features and TF-IDF edge embeddings.

        :param graph: DGLGraph
        :return: word embeddings tensor [num_word_nodes, embed_size]
        """
        # Get word node IDs
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        # Get edges from word to sentence/doc
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)
        # Get word IDs
        wid = graph.nodes[wnode_id].data["id"]
        # Lookup word embeddings
        w_embed = self._embed(wid)
        # Store embeddings in graph
        graph.nodes[wnode_id].data["embed"] = w_embed
        # TF-IDF embedding for edges
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        return w_embed

    def set_snfeature(self, graph):
        """
        Compute and combine CNN and LSTM features for sentence nodes.

        :param graph: DGLGraph
        :return: combined sentence features [num_sentence_nodes, n_feature_size * 2]
        """
        # Get sentence node IDs
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        # CNN features
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        # Extract sentence embeddings for LSTM
        features, glen = self.get_snode_feat(graph, feat="sent_embedding")
        # LSTM features
        lstm_feature = self._sent_lstm_feature(features, glen)
        # Concatenate CNN and LSTM features
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)
        return node_feature

    @staticmethod
    def get_snode_feat(G, feat):
        """
        Extract sentence node features from a batched graph.

        :param G: DGLGraph (batched)
        :param feat: feature name in node data
        :return: features list, node counts list
        """
        # Split batched graph into individual graphs
        glist = dgl.unbatch(G)
        feature = []
        glen = []
        for g in glist:
            # Filter sentence nodes
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            feature.append(g.nodes[snode_id].data[feat])
            glen.append(len(snode_id))
        return feature, glen
