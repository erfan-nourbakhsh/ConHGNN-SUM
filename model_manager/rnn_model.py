import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph


class RnnHSGModel(nn.Module):
    """
    Hierarchical Summarization Graph Model with RNN layer for sentence-level processing.

    Combines the HSumGraph (graph-based sentence embeddings) with a bidirectional GRU to capture
    sequential dependencies between sentences and finally classifies each sentence.
    """

    def __init__(self, hps, embed):
        """
        Initialize the RnnHSGModel.

        :param hps: Hyperparameter object containing model settings
        :param embed: Pretrained word embedding tensor or nn.Embedding
        """
        super(RnnHSGModel, self).__init__()

        self.hps = hps  # store hyperparameters
        self.HSG = HSumGraph(embed=embed, hps=hps)  # instantiate the graph-based sentence encoder

        # Bidirectional GRU for sentence-level sequential modeling
        # input_size = hidden_size from HSG output, hidden_size = same, 4 layers, batch_first=True
        self.rnn = torch.nn.GRU(
            input_size=hps.hidden_size,
            hidden_size=hps.hidden_size,
            num_layers=4,
            bias=True,
            batch_first=True,
            bidirectional=True
        )

        # Classifier: maps RNN output to 2-class prediction per sentence
        self.classifier = nn.Sequential(
            nn.Linear(2 * hps.hidden_size, hps.hidden_size),  # project bi-directional hidden to hidden_size
            nn.LeakyReLU(),                                   # non-linearity
            nn.Linear(hps.hidden_size, 2),                   # final 2-class output
            nn.ELU()                                         # final activation
        )

        # Move model parameters to the device specified in hyperparameters
        self.to(hps.device)

    def forward(self, graph):
        """
        Forward pass of the model.

        :param graph: Batched DGLGraph containing word and sentence nodes.
                      Sentence nodes must have dtype=1.
        :return: Tensor of shape [total_sentences, 2], classification scores for each sentence
        """
        # Step 1: Compute sentence-level embeddings using HSumGraph
        sent_features = self.HSG(graph)  # shape [total_sentences, hidden_size]

        # Step 2: Unbatch the graph to get individual graphs in the batch
        graph_list = dgl.unbatch(graph)

        # Step 3: Get number of sentences per graph (needed for RNN splitting)
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]

        # Step 4: Initialize empty tensor for concatenating RNN outputs
        rnn_results = torch.Tensor().to(self.hps.device)

        # Step 5: Process each graph's sentence embeddings sequentially with RNN
        for sentence_vector in torch.split(sent_features, indices):
            # sentence_vector shape: [num_sentences_in_graph, hidden_size]
            rnn_result = self.rnn(sentence_vector.unsqueeze(dim=0))[0].squeeze()  # add batch dim, then remove
            # Concatenate results across the batch
            rnn_results = torch.cat([rnn_results, rnn_result])

        # Step 6: Apply classifier on concatenated RNN outputs
        result = self.classifier(rnn_results)

        # Step 7: Return classification scores
        return result
        # Optional: could also return intermediate probabilities from HSG if needed
