import torch
import torch.nn as nn

from module.GATLayer import SGATLayer

######################################### StackLayer #########################################

class MultiHeadSGATLayer(nn.Module):
    """
    Multi-Head Sentence-to-Sentence Graph Attention Layer (SGAT) 
    that combines multiple SGAT layers in parallel (multi-head attention).
    """
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):
        """
        :param in_dim: int, input feature dimension
        :param out_dim: int, output feature dimension per head
        :param num_heads: int, number of attention heads
        :param attn_drop_out: float, dropout probability on input features for each head
        :param merge: str, 'cat' to concatenate heads, 'mean' to average heads
        """
        super(MultiHeadSGATLayer, self).__init__()
        self.heads = nn.ModuleList()  # store all attention heads
        for i in range(num_heads):
            self.heads.append(
                SGATLayer(in_dim, out_dim)  # each head is an independent SGAT layer
            )
        self.merge = merge              # how to merge head outputs
        self.dropout = nn.Dropout(attn_drop_out)  # dropout applied to inputs of each head

    def forward(self, g, h):
        """
        Forward pass through multi-head SGAT.
        :param g: DGLGraph
        :param h: node features [n_nodes, in_dim]
        :return: aggregated node features [n_nodes, out_dim * num_heads] if concat, else [n_nodes, out_dim]
        """
        # Apply each attention head to the input, with dropout
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # list of [n_nodes, out_dim]
        if self.merge == 'cat':
            # concatenate head outputs along feature dimension
            return torch.cat(head_outs, dim=1)  # [n_nodes, out_dim * num_heads]
        else:
            # merge by averaging across heads
            return torch.mean(torch.stack(head_outs), dim=0)  # [n_nodes, out_dim]


class MultiHeadLayer(nn.Module):
    """
    Generic Multi-Head Graph Attention Layer supporting any given GAT-type layer.
    Can be used for Word-to-Sentence or Sentence-to-Word attention layers.
    """
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, feat_embed_size, layer, merge='cat'):
        """
        :param in_dim: int, input feature dimension
        :param out_dim: int, output feature dimension per head
        :param num_heads: int, number of attention heads
        :param attn_drop_out: float, dropout probability for each head input
        :param feat_embed_size: int, edge or auxiliary feature size (for WSGAT/SWGAT)
        :param layer: class, the attention layer to use (WSGATLayer, SWGATLayer, etc.)
        :param merge: str, 'cat' to concatenate, 'mean' to average
        """
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()  # store all attention heads
        for i in range(num_heads):
            # instantiate a separate layer for each head
            self.heads.append(layer(in_dim, out_dim, feat_embed_size))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)  # dropout applied to input features

    def forward(self, g, h):
        """
        Forward pass through multi-head generic GAT layer.
        :param g: DGLGraph
        :param h: node features [n_nodes, in_dim]
        :return: merged node features [n_nodes, out_dim*num_heads] if concat, else [n_nodes, out_dim]
        """
        # Apply each attention head
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # list of [n_nodes, out_dim]
        if self.merge == 'cat':
            # concatenate head outputs along feature dimension
            result = torch.cat(head_outs, dim=1)  # [n_nodes, out_dim*num_heads]
        else:
            # merge by averaging
            result = torch.mean(torch.stack(head_outs), dim=0)  # [n_nodes, out_dim]
        return result
