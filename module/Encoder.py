from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from module.PositionEmbedding import get_sinusoid_encoding_table

WORD_PAD = "[PAD]"


class sentEncoder(nn.Module):
    """
    Sentence Encoder using word embeddings, positional embeddings, and CNNs with multiple kernel sizes.
    Encodes a sentence into a fixed-size embedding.
    """

    def __init__(self, hps, embed):
        """
        Initialize the sentence encoder.

        :param hps: object containing hyperparameters
                    - word_emb_dim: int, word embedding dimension
                    - sent_max_len: int, maximum number of tokens in a sentence
                    - word_embedding: bool, whether to use word embeddings
                    - embed_train: bool, whether to train word embeddings
                    - cuda: bool, whether to use GPU
        :param embed: nn.Embedding, pretrained word embedding
        """
        super(sentEncoder, self).__init__()

        # store hyperparameters
        self._hps = hps
        self.sent_max_len = hps.sent_max_len
        embed_size = hps.word_emb_dim  # word embedding dimension

        # CNN hyperparameters
        input_channels = 1          # 1 channel for word embeddings
        out_channels = 50           # number of filters per kernel size
        min_kernel_size = 2         # smallest convolution kernel height
        max_kernel_size = 7         # largest convolution kernel height
        width = embed_size          # width = embedding dimension

        # word embedding layer
        self.embed = embed

        # positional embedding layer (sinusoidal encoding)
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.sent_max_len + 1, embed_size, padding_idx=0), freeze=True)

        # CNN layers with different kernel heights
        self.convs = nn.ModuleList([
            nn.Conv2d(input_channels, out_channels, kernel_size=(height, width))
            for height in range(min_kernel_size, max_kernel_size + 1)
        ])

        # initialize CNN weights using Xavier normal initialization
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))

    def forward(self, input):
        """
        Forward pass for sentence encoder.

        :param input: torch.Tensor [s_nodes, seq_len], batch of sentences as token IDs
        :return: torch.Tensor [s_nodes, out_channels * num_kernel_sizes], sentence embeddings
        """
        # calculate sentence length per example (ignoring padding)
        input_sent_len = ((input != 0).sum(dim=1)).int()  # [s_nodes]

        # get word embeddings for input tokens
        enc_embed_input = self.embed(input)  # [s_nodes, seq_len, embed_dim]

        # build positional indices for each token in sentence
        sent_pos_list = []
        for sentlen in input_sent_len:
            # create position list from 1 to sentence length
            sent_pos = list(range(1, min(self.sent_max_len, sentlen) + 1))
            # pad positions with zeros for shorter sentences
            sent_pos.extend([0] * int(self.sent_max_len - sentlen))
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long()  # [s_nodes, sent_max_len]

        # move position indices to GPU if CUDA is enabled
        if self._hps.cuda:
            input_pos = input_pos.cuda()

        # get positional embeddings
        enc_pos_embed_input = self.position_embedding(input_pos.long())  # [s_nodes, seq_len, embed_dim]

        # sum word embeddings and positional embeddings
        enc_conv_input = enc_embed_input + enc_pos_embed_input  # [s_nodes, seq_len, embed_dim]

        # add channel dimension for CNN input
        enc_conv_input = enc_conv_input.unsqueeze(1)  # [s_nodes, 1, seq_len, embed_dim]

        # apply convolution + ReLU for each kernel size
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs]
        # each element: [s_nodes, out_channels, seq_len - kernel_height + 1]

        # max-pooling over sequence length for each kernel
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output]
        # each element: [s_nodes, out_channels]

        # concatenate all kernel outputs to get final sentence embedding
        sent_embedding = torch.cat(enc_maxpool_output, 1)  # [s_nodes, out_channels * num_kernel_sizes]

        return sent_embedding  # [s_nodes, 50 * 6 = 300 if 6 kernel sizes]
