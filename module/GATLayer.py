import torch
import torch.nn as nn
import torch.nn.functional as F

######################################### SubLayer #########################################
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used in GAT/Transformer architectures.
    Applies two linear transformations with ReLU in between, followed by residual connection
    and layer normalization.
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        :param d_in: int, input feature dimension
        :param d_hid: int, hidden feature dimension
        :param dropout: float, dropout probability
        """
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # first linear layer, applied position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # second linear layer, position-wise
        self.layer_norm = nn.LayerNorm(d_in)  # layer normalization for stability
        self.dropout = nn.Dropout(dropout)    # dropout to prevent overfitting

    def forward(self, x):
        """
        Forward pass of FFN.
        :param x: [batch_size, seq_len, d_in] input features
        :return: [batch_size, seq_len, d_in] output features
        """
        assert not torch.any(torch.isnan(x)), "FFN input contains NaN"
        residual = x  # store original input for residual connection
        output = x.transpose(1, 2)  # convert to [batch_size, d_in, seq_len] for Conv1d
        output = self.w_2(F.relu(self.w_1(output)))  # apply two conv layers with ReLU
        output = output.transpose(1, 2)  # convert back to [batch_size, seq_len, d_in]
        output = self.dropout(output)    # apply dropout
        output = self.layer_norm(output + residual)  # residual connection + layer norm
        assert not torch.any(torch.isnan(output)), "FFN output contains NaN"
        return output


######################################### HierLayer #########################################

class SGATLayer(nn.Module):
    """
    Sentence-to-Sentence Graph Attention Layer (SGAT).
    Updates sentence nodes based on neighboring sentence nodes using attention.
    """
    def __init__(self, in_dim, out_dim, weight=0):
        super(SGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)      # feature transformation
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)  # attention computation

    def edge_attention(self, edges):
        """
        Compute attention score for each edge.
        :param edges: edge batch
        :return: dict with attention 'e'
        """
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # concat source & target features
        wa = F.leaky_relu(self.attn_fc(z2))                      # apply leaky ReLU to attention score
        return {'e': wa}

    def message_func(self, edges):
        """Pass source features and attention scores to target nodes."""
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """Aggregate messages using attention weights."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)  # normalize attention
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)  # weighted sum
        return {'sh': h}

    def forward(self, g, h):
        """
        Forward pass for SGAT.
        :param g: DGLGraph
        :param h: node features
        :return: updated sentence node features
        """
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)  # sentence nodes
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 0)  # sentence-sentence edges
        z = self.fc(h)  # transform node features
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)  # compute attention scores
        g.pull(snode_id, self.message_func, self.reduce_func)  # aggregate messages
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class WSGATLayer(nn.Module):
    """
    Word-to-Sentence Graph Attention Layer (WSGAT).
    Updates sentence nodes based on neighboring word nodes and edge features.
    """
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)          # transform node features
        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)  # transform edge features
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)      # attention layer

    def edge_attention(self, edges):
        """Compute attention score for edges based on node + edge features."""
        dfeat = self.feat_fc(edges.data["tfidfembed"])                  # edge feature
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # concat src, dst, edge
        wa = F.leaky_relu(self.attn_fc(z2))                              # attention score
        return {'e': wa}

    def message_func(self, edges):
        """Pass source features and attention scores to target nodes."""
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """Aggregate messages using attention weights."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        """
        Forward pass for WSGAT.
        :param g: DGLGraph
        :param h: node features
        :return: updated sentence node features
        """
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)  # word nodes
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)  # sentence nodes
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))  # edges
        z = self.fc(h)  # transform word node features
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=wsedge_id)  # compute edge attention
        g.pull(snode_id, self.message_func, self.reduce_func)  # aggregate to sentence nodes
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class SWGATLayer(nn.Module):
    """
    Sentence-to-Word Graph Attention Layer (SWGAT).
    Updates word nodes based on neighboring sentence nodes and edge features.
    """
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)        # node feature transform
        self.feat_fc = nn.Linear(feat_embed_size, out_dim)      # edge feature transform
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)   # attention layer

    def edge_attention(self, edges):
        """Compute attention score for edges."""
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # edge feature
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # concat src, dst, edge
        wa = F.leaky_relu(self.attn_fc(z2))                              # attention score
        return {'e': wa}

    def message_func(self, edges):
        """Pass features and attention to target nodes."""
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """Aggregate messages with attention weights."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        """
        Forward pass for SWGAT.
        :param g: DGLGraph
        :param h: node features
        :return: updated word node features
        """
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)  # word nodes
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)  # sentence nodes
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))  # edges
        z = self.fc(h)  # transform sentence node features
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)  # compute attention
        g.pull(wnode_id, self.message_func, self.reduce_func)  # aggregate to word nodes
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]
