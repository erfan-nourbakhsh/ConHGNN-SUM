import multiprocessing
import re
import os
import shutil
import copy
import datetime
import numpy as np
import torch

from rouge import Rouge
from .logger import *  # imports a custom logger
import sys

# Increase recursion limit to handle deep recursive calls (used in some evaluation functions)
sys.setrecursionlimit(10000)

# Paths for PyRouge evaluation
_ROUGE_PATH = "/remote-home/erfan-nourbakhsh/ROUGE/RELEASE-1.5.5"
_PYROUGE_TEMP_FILE = "/remote-home/erfan-nourbakhsh/"

# Mapping special tokens to readable punctuation
REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    """
    Cleans input text by converting to lowercase and replacing special tokens with readable symbols.

    :param x: str, input text
    :return: str, cleaned text
    """
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def rouge_eval(hyps, refer):
    """
    Compute mean ROUGE score (ROUGE-1, ROUGE-2, ROUGE-L) for a single pair of hypothesis and reference.

    :param hyps: str, predicted summary
    :param refer: str, reference summary
    :return: float, mean ROUGE F1 score
    """
    rouge = Rouge()
    try:
        score = rouge.get_scores(hyps, refer)[0]
        mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
    except:
        mean_score = 0.0
    return mean_score


def rouge_all(hyps, refer):
    """
    Returns detailed ROUGE scores for a single hypothesis-reference pair.

    :param hyps: str, predicted summary
    :param refer: str, reference summary
    :return: dict, ROUGE-1, ROUGE-2, ROUGE-L scores with precision, recall, f1
    """
    rouge = Rouge()
    score = rouge.get_scores(hyps, refer)[0]
    return score


def eval_label(match_true, pred, true, total, match):
    """
    Evaluate metrics like accuracy, precision, recall, F1 for predicted vs. true labels.

    :param match_true: tensor, correct predictions overlapping with ground truth
    :param pred: tensor, number of predicted positive samples
    :param true: tensor, number of true positive samples
    :param total: tensor, total samples
    :param match: tensor, matched samples
    :return: tuple of floats (accuracy, precision, recall, F1)
    """
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def pyrouge_score(hyps, refer, remap=True):
    """
    Wrapper for single hypothesis-reference ROUGE evaluation using PyRouge.

    :param hyps: str, predicted summary
    :param refer: str, reference summary
    :param remap: bool, whether to apply token remapping
    :return: dict, ROUGE scores
    """
    return pyrouge_score_all([hyps], [refer], remap)


def pyrouge_score_all(hyps_list, refer_list, remap=True):
    """
    Compute ROUGE scores using PyRouge for lists of hypotheses and references.

    :param hyps_list: list of str, predicted summaries
    :param refer_list: list of str, reference summaries
    :param remap: bool, whether to apply token remapping
    :return: dict, ROUGE scores
    """
    from pyrouge import Rouge155

    # Create temporary directories for PyRouge evaluation
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_FILE, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'result')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)

    # Write hypothesis and reference files
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

    # Initialize PyRouge
    r = Rouge155(_ROUGE_PATH)
    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    try:
        output = r.convert_and_evaluate(
            rouge_args="-e %s -a -m -n 2 -d" % os.path.join(_ROUGE_PATH, "data")
        )
        output_dict = r.output_to_dict(output)
    finally:
        logger.error("[ERROR] Error stop, delete PYROUGE_ROOT...")
        shutil.rmtree(PYROUGE_ROOT)

    # Format the ROUGE scores
    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = \
        output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = \
        output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = \
        output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']

    return scores


def cal_label(article, abstract):
    """
    Selects the best sentences from article to maximize ROUGE score with reference abstract.

    :param article: list of str, sentences in the article
    :param abstract: str, reference summary
    :return: list of int, indices of selected sentences
    """
    hyps_list = article
    refer = abstract
    scores = []

    # Compute ROUGE score for each sentence individually
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    # Select sentence with highest ROUGE
    selected = [int(np.argmax(scores))]
    selected_sent_cnt = 1
    best_rouge = np.max(scores)

    # Iteratively select sentences that maximize cumulative ROUGE
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break

    return selected


def _save_model(model, save_file):
    """
    Save model state to a file (blocking operation).

    :param model: torch.nn.Module, model to save
    :param save_file: str, path to save the model
    """
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def save_model(model, save_file):
    """
    Save model state in a separate process to avoid blocking main thread.

    :param model: torch.nn.Module, model to save
    :param save_file: str, path to save the model
    """
    multiprocessing.Process(target=_save_model, args=(model, save_file))
