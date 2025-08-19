import argparse
import datetime
import os
import time
import sys
import logging

import torch
from rouge import Rouge

from model_manager.model import Model
from Tester import SLTester
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools import utils
from tools.logger import *
from config import pars_args
from utils import set_device
from data_manager import data_loaders


def load_test_model(model, model_name, eval_dir, save_root):
    """
    Load a trained model for evaluation.

    Args:
        model (torch.nn.Module): Model instance.
        model_name (str): Name of the checkpoint file to load.
        eval_dir (str): Directory for evaluation checkpoints.
        save_root (str): Root directory where checkpoints are stored.

    Returns:
        torch.nn.Module: Loaded model ready for evaluation.
    """
    # Construct full path to the model checkpoint
    path = os.path.join(save_root, model_name)
    # Load model weights
    model.load_state_dict(torch.load(path))
    return model

    # The following code is unreachable due to the return above, but describes
    # alternative loading logic if needed
    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("Invalid model name")
    # Check if checkpoint exists
    if not os.path.exists(bestmodel_load_path):
        logger.error("[ERROR] Restoring %s for testing... Path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing from %s", model_name, bestmodel_load_path)
    model.load_state_dict(torch.load(bestmodel_load_path))
    return model


def run_test(model, dataset, loader, model_name, hps):
    """
    Run evaluation on the test set using the provided model.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (Dataset): Dataset object.
        loader (DataLoader): PyTorch DataLoader for test set.
        model_name (str): Name of the model checkpoint.
        hps (argparse.Namespace): Hyperparameters and settings.
    """
    # Directories for test results and evaluation
    test_dir = os.path.join(hps.save_root, "test")
    eval_dir = os.path.join(hps.save_root, "eval")

    # Ensure test directory exists
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Ensure eval directory exists; otherwise raise exception
    if not os.path.exists(eval_dir):
        logger.exception("[Error] eval_dir %s doesn't exist.", eval_dir)
        raise Exception(f"[Error] eval_dir {eval_dir} doesn't exist. Run in train mode to create it.")

    # Load the model checkpoint
    model = load_test_model(model, model_name, eval_dir, hps.save_root)
    model.eval()  # Set model to evaluation mode

    iter_start_time = time.time()  # Start timer

    # Disable gradient computation for evaluation
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")
        tester = SLTester(model, hps.m, limited=hps.limited, test_dir=test_dir)

        # Iterate over test batches
        for i, (G, index) in enumerate(loader):
            G = G.to(hps.device)  # Move graph to device (CPU/GPU)
            tester.evaluation(G, index, dataset, blocking=hps.blocking)

    running_avg_loss = tester.running_avg_loss  # Average loss over all batches

    # Logging test statistics
    logger.info("The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error("During testing, no hypotheses were selected!")
        sys.exit(1)

    # Compute ROUGE scores
    if hps.use_pyrouge:
        if isinstance(tester.refer[0], list):
            logger.info("Multi-reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer)
        else:
            scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer)
    else:
        rouge = Rouge()
        scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    # Format ROUGE scores for logging
    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    # Save metrics and decoded files
    tester.getMetric()
    tester.SaveDecodeFile()

    # Log end of test
    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format(
        (time.time() - iter_start_time), float(running_avg_loss)))


def main():
    """
    Main function to run test evaluation for ConHGNN-SUM.
    It sets up environment, loads model, vocab, embeddings, and runs tests.
    """
    args = pars_args()  # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ConHGNN-SUM Model')

    # Set GPU devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    hps = args
    hps = set_device(hps=hps)  # Set device for computation (CPU/GPU)

    torch.set_printoptions(threshold=50000)  # For printing tensors without truncation

    # File paths for test data and resources
    DATA_FILE = os.path.join(args.data_dir, "test.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root

    # Setup logging to file
    if not os.path.exists(LOG_PATH):
        logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "test_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log environment info
    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)

    # Load vocabulary
    vocab = Vocab(VOCAL_FILE, args.vocab_size)

    # Initialize embeddings
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    logger.info(hps)  # Log hyperparameters

    # Test data paths
    test_w2s_path = os.path.join(args.cache_dir, "test.w2s.tfidf.jsonl")

    # Model selection
    if hps.model == "HSG":
        model = Model(hps, embed)
        logger.info("[MODEL] ConHGNN-SUM ")

        # Create test dataloader
        loader = data_loaders.make_dataloader(
            data_file=DATA_FILE, vocab=vocab, hps=hps, filter_word=FILTER_WORD, w2s_path=test_w2s_path,
            graphs_dir=os.path.join(args.cache_dir, "graphs/test")
        )

        if hps.fill_graph_cache:
            return

    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    # Set device for evaluation
    if args.cuda:
        hps.device = torch.device("cuda:0")
        logger.info("[INFO] Use cuda")
    else:
        hps.device = torch.device("cpu")
        logger.info("[INFO] Use CPU")

    # Run evaluation
    logger.info("[INFO] Decoding...")
    if hps.test_model == "multi":
        for i in range(3):
            model_name = "evalbestmodel_%d" % i
            run_test(model, loader.dataset, loader, model_name, hps)
    else:
        print(model)
        run_test(model, loader.dataset, loader, hps.test_model, hps)


if __name__ == '__main__':
    main()  # Entry point
