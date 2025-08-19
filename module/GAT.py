import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GATStackLayer import MultiHeadSGATLayer, MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward, WSGATLayer, SWGATLayer


######################################### SubModule #########################################
class WSWGAT(nn.Module):
    """
    Word-Sentence / Sentence-Word / Sentence-Sentence Graph Attention Network (GAT) module.

    This module applies a multi-head GAT layer followed by a position-wise feed-forward network.
    It supports three types of GAT:
        - W2S: Word to Sentence
        - S2W: Sentence to Word
        - S2S: Sentence to Sentence
    """

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out,
                 feat_embed_size, layerType):
        """
        Initialize the WSWGAT module.

        :param in_dim: int, input feature dimension
        :param out_dim: int, output feature dimension
        :param num_heads: int, number of attention heads
        :param attn_drop_out: float, attention dropout probability
        :param ffn_inner_hidden_size: int, hidden size for feed-forward layer
        :param ffn_drop_out: float, dropout for feed-forward layer
        :param feat_embed_size: int, feature embedding size
        :param layerType: str, type of GAT layer ("W2S", "S2W", or "S2S")
        """
        super().__init__()

        self.layerType = layerType  # store layer type

        # choose the GAT layer type based on the specified connection
        if layerType == "W2S":
            # Word -> Sentence layer
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size,
                                        layer=WSGATLayer)
        elif layerType == "S2W":
            # Sentence -> Word layer
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size,
                                        layer=SWGATLayer)
        elif layerType == "S2S":
            # Sentence -> Sentence layer
            self.layer = MultiHeadSGATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        # Position-wise Feed-Forward Network after GAT layer
        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s):
        """
        Forward pass of the WSWGAT module.

        :param g: DGLGraph, input graph
        :param w: torch.Tensor, word node features
        :param s: torch.Tensor, sentence node features
        :return: torch.Tensor, updated node features after GAT and feed-forward layers
        """
        # determine origin and neighbor nodes based on the layer type
        if self.layerType == "W2S":
            origin, neighbor = s, w  # update sentence nodes using word features
        elif self.layerType == "S2W":
            origin, neighbor = w, s  # update word nodes using sentence features
        elif self.layerType == "S2S":
            assert torch.equal(w, s)  # ensure input features are identical for S2S
            origin, neighbor = w, s
        else:
            origin, neighbor = None, None  # fallback (should not occur)

        # apply GAT layer with ELU activation
        h = F.elu(self.layer(g, neighbor))  # [num_nodes, out_dim]

        # add residual connection (original node features)
        h = h + origin

        # apply position-wise feed-forward network
        h = self.ffn(h.unsqueeze(0)).squeeze(0)  # [num_nodes, out_dim]

        return h
