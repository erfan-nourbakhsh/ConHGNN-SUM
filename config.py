import argparse

# Debug flag for optional debugging
_DEBUG_FLAG_ = False

def pars_args():
    """
    Parse command-line arguments for the ConHGNN-SUM model.

    Returns:
        argparse.Namespace: Parsed arguments object containing dataset paths,
                            model configurations, hyperparameters, and training options.
    """

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='ConHGNN-SUM Model')

    # Base root path for dataset and embeddings (changeable as per your system)
    root = "/"

    # -------------------- Dataset & Cache Paths -------------------- #
    parser.add_argument('--data_dir', type=str,
                        default=f'{root}\datasets\cnndm',
                        help='Directory where the dataset is located.')
    parser.add_argument('--cache_dir', type=str,
                        default=f'{root}/cache/CNNDM',
                        help='Directory to store processed dataset cache.')

    parser.add_argument('--embedding_path', type=str,
                        default=f'{root}\embeddings\glove.42B.300d.txt',
                        help='Path to pre-trained external word embeddings.')

    # -------------------- Model Selection & Testing -------------------- #
    parser.add_argument('--model', type=str, default='HSG', help='Model structure to use [HSG|HDSG]')
    parser.add_argument('--test_model', type=str,
                        default=r'/save/eval/bestmodel_0',
                        help='Select model checkpoint for testing [multi/evalbestmodel/trainbestmodel/earlystop]')

    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='Whether to use PyRouge for evaluation.')

    parser.add_argument('--restore_model', type=str, default='None',
                        help='Restore model for further training [bestmodel/bestFmodel/earlystop/None]')

    # -------------------- Output Paths -------------------- #
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory to save all model outputs.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for logging outputs.')

    # -------------------- General Hyperparameters -------------------- #
    parser.add_argument('--seed', type=int, default=666, help='Random seed [default: 666]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use GPU (CUDA) [default: True]')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Maximum vocabulary size [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size [default: 64]')
    parser.add_argument('--n_iter', type=int, default=1, help='Number of iteration hops [default: 1]')

    # -------------------- Word Embedding -------------------- #
    parser.add_argument('--word_embedding', action='store_true', default=False,
                        help='Use word embeddings [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Dimension of word embeddings [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False,
                        help='Train word embeddings during training [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='Feature embedding size [default: 50]')

    # -------------------- Model Layers -------------------- #
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='LSTM hidden state size [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='Node feature size [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size [default: 64]')
    parser.add_argument('--gcn_hidden_size', type=int, default=128, help='GCN hidden size [default: 128]')

    # -------------------- Transformer / Attention Parameters -------------------- #
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,
                        help='Positionwise FeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='Recurrent dropout probability [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='Attention dropout probability [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='FeedForward dropout probability [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,
                        help='Use orthonormal initialization for LSTM [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100, help='Maximum sentence length [default: 100]')
    parser.add_argument('--doc_max_timesteps', type=int, default=50, help='Maximum number of document timesteps [default: 50]')

    # -------------------- Training Settings -------------------- #
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate [default: 0.0005]')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='Use learning rate descent schedule')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='Enable gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping [default: 1.0]')

    parser.add_argument('-m', type=int, default=3, help='Decoded summary length')
    parser.add_argument('--save_label', action='store_true', default=True, help='Whether to save attention labels')

    parser.add_argument('--limited', action='store_true', default=False, help='Limit hypothesis length')
    parser.add_argument('--blocking', action='store_true', default=False, help='Use n-gram blocking')

    parser.add_argument('--max_instances', type=int, default=None, help='Maximum number of instances to use')
    parser.add_argument('--from_instances_index', type=int, default=0, help='Start index of instances to use')

    parser.add_argument('--use_cache_graph', type=bool, default=True, help='Use cached graph if available')
    parser.add_argument('--fill_graph_cache', type=bool, default=False, help='Fill graph cache during processing')

    # Parse and return all arguments
    args = parser.parse_args()
    return args
