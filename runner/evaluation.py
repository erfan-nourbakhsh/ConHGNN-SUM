import datetime
import os
import shutil
import time
import random
import numpy as np
import torch
from rouge import Rouge
from Tester import SLTester
from tools.logger import *


def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo):
    """
    Runs a full evaluation of the model on a validation dataset.
    Logs results, computes Rouge scores, and saves the best models according to loss or F-score.
    
    :param model: PyTorch model to evaluate
    :param loader: DataLoader for validation dataset
    :param valset: Validation dataset (text and reference summary)
    :param hps: hyperparameters object containing model and training configs
    :param best_loss: current best validation loss
    :param best_F: current best F-score
    :param non_descent_cnt: number of consecutive epochs with no improvement (used for early stopping)
    :param saveNo: number of saved models (used to rotate checkpoints)
    :return: updated best_loss, best_F, non_descent_cnt, saveNo
    """
    
    logger.info("[INFO] Starting eval for this model ...")
    
    # Create evaluation directory under the model save root
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Set model to evaluation mode (disables dropout, batchnorm, etc.)
    model.eval()

    iter_start_time = time.time()  # record start time

    with torch.no_grad():  # disables gradient computation
        tester = SLTester(model, hps.m)  # initialize tester with model and metrics
        for i, (G, index) in enumerate(loader):
            G = G.to(hps.device)  # move graph data to GPU if needed
            tester.evaluation(G, index, valset)  # evaluate batch

    running_avg_loss = tester.running_avg_loss  # get average loss over validation dataset

    # Safety check: ensure there are predictions and references
    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error("During testing, no hyps is selected!")
        return

    # Compute Rouge scores for all predictions
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    # Log time taken and validation loss
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format(
        (time.time() - iter_start_time), float(running_avg_loss)
    ))

    # Log Rouge scores
    log_score(scores_all=scores_all)

    # Compute F-score metric for model performance
    tester.getMetric()
    F = tester.labelMetric

    # Save model if it has achieved a new best validation loss
    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (saveNo % 3))  # rotate over 3 checkpoints

        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path
            )
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path
            )

        # Save model checkpoint
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)

        best_loss = running_avg_loss
        non_descent_cnt = 0  # reset non-descent counter
        saveNo += 1  # increment saved model counter
    else:
        non_descent_cnt += 1  # no improvement in this epoch

    # Save model if it has achieved a new best F-score
    if best_F is None or best_F < F:
        bestmodel_save_path = os.path.join(eval_dir, 'HSGmodel')  # path for best-F model

        if best_F is not None:
            logger.info(
                '[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s',
                float(F), float(best_F), bestmodel_save_path
            )
        else:
            logger.info(
                '[INFO] Found new best model with %.6f F. The original F is None, Saving to %s',
                float(F), bestmodel_save_path
            )

        # Save model checkpoint
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)

        best_F = F

    # Return updated metrics and counters
    return best_loss, best_F, non_descent_cnt, saveNo
