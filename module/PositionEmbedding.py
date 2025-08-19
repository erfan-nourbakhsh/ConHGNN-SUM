import torch
import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    Generate a sinusoidal positional encoding table as used in Transformers.
    Each position is encoded with sine and cosine functions of different frequencies.
    
    :param n_position: int, number of positions (sequence length)
    :param d_hid: int, hidden dimension (embedding size)
    :param padding_idx: int or None, index of padding token to be zeroed out
    :return: torch.FloatTensor of shape [n_position, d_hid]
    """
    
    def cal_angle(position, hid_idx):
        """
        Compute the angle rate for a given position and hidden dimension index.
        Formula: pos / 10000^(2i/d_hid) for dimension 2i and 2i+1
        """
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        """
        Get the vector of angles for all hidden dimensions at a given position.
        :param position: int, position index
        :return: list of floats, size d_hid
        """
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    # build the positional encoding table for all positions
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])  # [n_position, d_hid]

    # apply sine to even indices (2i)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # apply cosine to odd indices (2i+1)
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # set the padding index row to zeros, if specified
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    # convert numpy array to torch FloatTensor
    return torch.FloatTensor(sinusoid_table)
