import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph
from torch.nn.functional import normalize


class TextRepresentationHSGModel(HSumGraph):
    """
    HSumGraph-based model that augments sentence embeddings with text-level LSTM representation.
    
    Inherits HSumGraph for hierarchical sentence representation.
    Uses an LSTM to capture sequential dependencies between sentences and a classifier for final predictions.
    """

    def __init__(self, hps, embed):
        """
        Initialize the model.

        :param hps: Hyperparameter object containing model settings
        :param embed: Pretrained word embedding tensor or nn.Embedding
        """
        # Initialize HSumGraph parent class
        super(TextRepresentationHSGModel, self).__init__(embed=embed, hps=hps)

        # LSTM for modeling sequential sentence embeddings
        self.rnn = nn.LSTM(hps.hidden_size, hps.hidden_size, 1, bias=True, batch_first=True)

        # Classifier combines graph-based embeddings and LSTM-derived text representation
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.n_feature, 64),  # project concatenated features to hidden size 64
            nn.ELU(),                           # activation
            nn.Linear(64, 2)                    # output 2-class prediction
        )

        # Move model to device
        self.to(hps.device)

    def forward(self, graph):
        """
        Forward pass of the model.

        :param graph: Batched DGLGraph containing word and sentence nodes
        :return: Tensor of shape [total_sentences, 2], sentence-level predictions
        """
        # Step 1: Initialize word node features
        word_feature = self.set_wnfeature(graph)  # [num_word_nodes, embed_size]

        # Step 2: Initialize sentence node features using CNN + LSTM projection
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [num_sentence_nodes, n_feature_size]

        # Step 3: Initialize graph propagation states
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)  # word->sentence attention

        # Step 4: Iterative message passing between sentences and words
        for i in range(self._n_iter):
            word_state = self.sent2word(graph, word_state, sent_state)  # sentence->word attention
            sent_state = self.word2sent(graph, word_state, sent_state)  # word->sentence attention

        # Step 5: Compute text-level LSTM representation per document
        i = 0
        text_feature = torch.Tensor().to(self._hps.device)
        for g in dgl.unbatch(graph):  # unbatch each document
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            sentence_len = len(snode_id)
            text_sent_feature = sent_feature[i:i + sentence_len, :]
            
            # Pass sentence embeddings through LSTM and repeat for all sentences
            text_feature = torch.cat([
                text_feature,
                self.rnn(text_sent_feature.unsqueeze(dim=0))[1][1].squeeze().repeat(
                    sentence_len).reshape((sentence_len, -1))
            ])
            i += sentence_len

        # Step 6: Concatenate HSG sentence embeddings and text-level embeddings, then classify
        result = self.classifier(torch.cat([sent_state, text_feature], 1))

        return result


class RnnTextRepresentationModel(nn.Module):
    """
    Model that combines HSumGraph sentence embeddings with a multi-layer LSTM text representation.
    Applies dropout and a classifier for sentence-level predictions.
    """

    def __init__(self, hps, embed):
        """
        Initialize the model.

        :param hps: Hyperparameter object containing model settings
        :param embed: Pretrained word embedding tensor or nn.Embedding
        """
        super(RnnTextRepresentationModel, self).__init__()

        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)  # graph-based sentence encoder

        # 2-layer LSTM for sequential modeling of sentence embeddings
        self.rnn_text_representation = nn.LSTM(hps.hidden_size, hps.hidden_size, 2, bias=True, batch_first=True)

        # Classifier combining sentence embeddings and text-level features
        self.classifier = nn.Sequential(
            nn.Linear(hps.hidden_size * 3, 64),  # concatenate sent_features + text_feature (hence *3)
            nn.ELU(),
            nn.Linear(64, 2),
            nn.ELU()
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Move model to device
        self.to(hps.device)

    def forward(self, graph):
        """
        Forward pass of the model.

        :param graph: Batched DGLGraph containing word and sentence nodes
        :return: Tensor of shape [total_sentences, 2], sentence-level predictions
        """
        # Step 1: Compute HSG-based sentence embeddings
        sent_features = self.HSG(graph)

        # Step 2: Unbatch the graph into individual documents
        graph_list = dgl.unbatch(graph)

        # Step 3: Determine number of sentences in each document
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]

        # Step 4: Initialize tensor for text-level features
        text_feature = torch.Tensor().to(self.hps.device)

        # Step 5: Compute text-level LSTM representation for each document
        for sentence_vector in torch.split(sent_features, indices):
            sentence_len = sentence_vector.shape[0]

            # LSTM hidden state for the document
            new_text_feature = self.rnn_text_representation(sentence_vector.unsqueeze(dim=0))[1][1].reshape(-1)

            # Repeat document-level representation for each sentence
            new_text_feature = new_text_feature.repeat(sentence_len).reshape((sentence_len, -1))

            # Concatenate text-level features across all documents
            text_feature = torch.cat([text_feature, new_text_feature])

        # Step 6: Concatenate sentence embeddings and text-level features, apply classifier + dropout
        result = self.dropout(self.classifier(torch.cat([sent_features, text_feature], 1)))

        return result
