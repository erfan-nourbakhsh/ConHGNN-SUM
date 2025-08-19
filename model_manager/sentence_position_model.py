import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph

# Predefined positional probabilities for sentences in a document
# Each entry represents the probability of being a key sentence based on its position
position_probabilities = [0.199290, 0.228100, 0.222377, 0.222499, 0.192609, 0.153802, 0.123783, 0.101765, 0.088789,
                          0.078482, 0.071153, 0.066298, 0.060599, 0.055165, 0.049167, 0.043374, 0.035558, 0.028654,
                          0.021482, 0.016149, 0.012366, 0.008329, 0.005747, 0.004382, 0.003083, 0.002205, 0.001501,
                          0.001024, 0.000731, 0.000470, 0.000293, 0.000136, 0.000104, 0.000098, 0.000052, 0.000014,
                          0.000010, 0.000014, 0.000003] + [0.000003] * 20

# Convert to tensor of shape [num_positions, 2] representing [non-key probability, key probability]
position_probabilities = torch.Tensor([[1 - x, x] for x in position_probabilities])


class RnnHSGModel(nn.Module):
    """
    Hierarchical Summarization Graph Model with LSTM layer and positional bias.

    Combines HSumGraph (graph-based sentence embeddings) with an LSTM for sequential modeling
    and incorporates positional probabilities to improve sentence-level key sentence prediction.
    """

    def __init__(self, hps, embed):
        """
        Initialize the model.

        :param hps: Hyperparameter object containing model settings
        :param embed: Pretrained word embedding tensor or nn.Embedding
        """
        super(RnnHSGModel, self).__init__()

        self.hps = hps  # store hyperparameters
        self.HSG = HSumGraph(embed=embed, hps=hps)  # graph-based sentence encoder

        # LSTM for sequential modeling of sentence embeddings
        # input_size = HSG hidden size, hidden_size = 128, 1 layer, default bias
        self.rnn = torch.nn.LSTM(hps.hidden_size, 128, 1, bias=True)

        # Classifier for LSTM output to 2-class sentence probability
        self.classifier = nn.Sequential(
            nn.Linear(128, 2),  # project LSTM hidden state to 2 classes
            nn.Sigmoid()         # squash to [0,1]
        )

        # Store predefined positional probabilities on the device
        self.position_probabilities = position_probabilities.to(hps.device)

        # Softmax function for normalizing outputs
        self.softmax = nn.Softmax(dim=-1)

        # Move entire model to device
        self.to(hps.device)

    def forward(self, graph):
        """
        Forward pass of the model.

        :param graph: Batched DGLGraph containing word and sentence nodes.
                      Sentence nodes must have dtype=1.
        :return: Tensor of shape [total_sentences, 2], combining HSG, LSTM, and positional predictions
        """

        # Step 1: Compute HSG-based sentence probabilities and embeddings
        hsg_p, sent_features = self.HSG(graph)  # hsg_p: HSG prediction, sent_features: embeddings

        # Step 2: Unbatch the graph to process documents individually
        graph_list = dgl.unbatch(graph)

        # Step 3: Count number of sentences per graph
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]

        # Step 4: Initialize empty tensors for concatenated LSTM results and positional probabilities
        rnn_results = torch.Tensor().to(self.hps.device)
        batch_position_probabilities = torch.Tensor().to(self.hps.device)

        # Step 5: Process each document's sentence embeddings
        for sentence_vector in torch.split(sent_features, indices):
            # Apply LSTM to the sentence embeddings (unsqueezed for batch dimension)
            new_rnn_result = self.classifier(self.rnn(sentence_vector)[0])

            # Concatenate LSTM outputs across all documents
            rnn_results = torch.cat([rnn_results, new_rnn_result])

            # Concatenate positional probabilities for sentences
            batch_position_probabilities = torch.cat(
                [batch_position_probabilities, self.position_probabilities[:sentence_vector.shape[0], :]]
            )

        # Step 6: Combine HSG predictions, positional bias, and LSTM predictions
        # Normalize HSG and LSTM predictions via softmax, sum with positional probabilities, divide by 3
        return (self.softmax(hsg_p) + batch_position_probabilities + self.softmax(rnn_results)) / 3
