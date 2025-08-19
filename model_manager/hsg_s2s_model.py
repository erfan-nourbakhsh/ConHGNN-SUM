import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv  # Graph Attention Network convolution
from model_manager.hsg_model import HSumGraph  # base hierarchical summarization model


class HSumGraphWithS2SModel(nn.Module):
    """
    Hierarchical Summarization Graph model with added sentence-to-sentence (S2S) GAT layer.
    
    - Uses HSumGraph to get initial sentence features from word-sentence graph.
    - Builds a fully-connected sentence graph (edges between all consecutive sentences).
    - Applies GATConv to propagate information between sentences.
    - Classifies each sentence (binary classification in this example).
    """
    
    def __init__(self, hps, embed):
        """
        Initializes the HSumGraphWithS2SModel.
        
        :param hps: hyperparameters object
        :param embed: torch.nn.Embedding, word embedding matrix
        """
        super(HSumGraphWithS2SModel, self).__init__()

        self.hps = hps  # store hyperparameters

        # Base hierarchical summarization graph model (word-sentence interactions)
        self.HSG = HSumGraph(embed=embed, hps=hps)

        # Number of attention heads for sentence-to-sentence GAT
        self.num_heads = 4

        # Sentence-to-sentence GAT layer
        self.s2s_gat_conv = GATConv(
            in_feats=hps.hidden_size,  # input dimension
            out_feats=hps.hidden_size,  # output dimension per head
            num_heads=self.num_heads   # number of attention heads
        )

        # Final classifier for sentence-level prediction
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hps.hidden_size * self.num_heads, 128),  # project concatenated heads
            torch.nn.LeakyReLU(),  # activation
            torch.nn.Linear(128, 2)  # binary classification
        )

        # Move model to device
        self.to(hps.device)

    def make_sentence_graph(self, graph):
        """
        Constructs a fully-connected sentence-to-sentence graph for each document in the batch.
        
        :param graph: batched DGLGraph with sentence nodes
        :return: DGLGraph representing sentence-to-sentence connections
        """
        # Initialize edge lists
        u, v = torch.Tensor([]), torch.Tensor([])  
        last_index = 0  # keeps track of node offset in batched graphs

        # Split batched graph into individual graphs
        graphs = dgl.unbatch(graph)

        # Iterate through each document graph
        for g in graphs:
            # Get sentence node IDs
            sentences = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

            # Create edges between consecutive sentences
            new_u = torch.Tensor(list(range(len(sentences) - 1))) + last_index
            new_v = torch.Tensor(list(range(1, len(sentences)))) + last_index

            # Add bidirectional edges
            u = torch.cat([u, new_u, new_v])
            v = torch.cat([v, new_v, new_u])

            # Update node offset for next document
            last_index += len(sentences)

        # Construct and return DGL graph
        return dgl.graph((list(u), list(v)))

    def forward(self, graph):
        """
        Forward pass for HSumGraphWithS2SModel.
        
        :param graph: DGLGraph (batched) with word and sentence nodes
        :return: tensor [num_sentence_nodes, 2] of predicted sentence labels
        """
        # Step 1: Get initial sentence features from HSumGraph (word-sentence interactions)
        sent_features = self.HSG(graph)

        # Step 2: Build sentence-to-sentence graph
        sentence_graph = self.make_sentence_graph(graph).to(self.hps.device)

        # Step 3: Apply GAT to propagate information between sentences
        sent_features = self.s2s_gat_conv(sentence_graph, sent_features)

        # Step 4: Flatten the multi-head outputs
        sent_features = sent_features.reshape(-1, self.num_heads * self.hps.hidden_size)

        # Step 5: Classify each sentence
        result = self.classifier(sent_features)

        return result
