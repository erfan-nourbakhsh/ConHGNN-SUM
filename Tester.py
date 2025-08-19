import torch  # PyTorch for tensor operations and deep learning
import dgl    # Deep Graph Library for graph neural networks
import os     # For file operations

from tools.utils import eval_label  # Custom utility for evaluation metrics
from tools.logger import *          # Custom logger for info/debug/error messages


class TestPipLine():
    """
    Base class for testing pipeline.
    Handles model evaluation, decoding, and metrics collection.

    Attributes:
        model: PyTorch model to evaluate.
        m: Number of sentences to select in summarization.
        test_dir: Directory to save decoded summaries.
        limited: Boolean flag to limit predictions to reference length.
    """
    def __init__(self, model, m, test_dir, limited):
        """
        Initialize the testing pipeline.
        """
        self.model = model
        self.limited = limited
        self.m = m
        self.test_dir = test_dir
        self.extracts = []  # List of extracted sentence indices

        self.batch_number = 0
        self.running_loss = 0
        self.example_num = 0
        self.total_sentence_num = 0

        self._hyps = []   # List of hypothesis summaries
        self._refer = []  # List of reference summaries

    def evaluation(self, G, index, valset):
        """
        Placeholder method to evaluate a batch.
        Should be implemented in subclass.
        """
        pass

    def getMetric(self):
        """
        Placeholder method to calculate metrics.
        Should be implemented in subclass.
        """
        pass

    def SaveDecodeFile(self):
        """
        Save all hypothesis and reference pairs into a file for later inspection.
        Each file is saved in a timestamped directory.
        """
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.test_dir, nowTime)
        with open(log_dir, "wb") as resfile:
            for i in range(self.rougePairNum):
                resfile.write(b"[Reference]\t")
                resfile.write(self._refer[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(self._hyps[i].encode('utf-8'))
                resfile.write(b"\n\n\n")

    @property
    def running_avg_loss(self):
        """Return the average loss over all batches."""
        return self.running_loss / self.batch_number

    @property
    def rougePairNum(self):
        """Return the number of hypothesis-reference pairs collected."""
        return len(self._hyps)

    @property
    def hyps(self):
        """
        Return hypothesis summaries.
        If limited=True, truncate each hypothesis to the same number of words as its reference.
        """
        if self.limited:
            hlist = []
            for i in range(self.rougePairNum):
                k = len(self._refer[i].split(" "))
                lh = " ".join(self._hyps[i].split(" ")[:k])
                hlist.append(lh)
            return hlist
        else:
            return self._hyps

    @property
    def refer(self):
        """Return the reference summaries."""
        return self._refer

    @property
    def extractLabel(self):
        """Return the list of extracted sentence indices."""
        return self.extracts


class SLTester(TestPipLine):
    """
    Sequence Labeling Tester for extractive summarization.

    Extends TestPipLine and implements actual evaluation.
    """
    def __init__(self, model, m, test_dir=None, limited=False, blocking_win=3):
        super().__init__(model, m, test_dir, limited)
        self.pred, self.true, self.match, self.match_true = 0, 0, 0, 0  # Counters for evaluation
        self._F = 0  # F-score
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Per-node loss
        self.blocking_win = blocking_win  # N-gram blocking window size

    def evaluation(self, G, index, dataset, blocking=False):
        """
        Evaluate a batch of graphs.

        Args:
            G: DGL batch graph
            index: List of example indices
            dataset: Dataset object providing text and reference summaries
            blocking: Boolean flag to apply n-gram blocking
        """
        self.batch_number += 1
        outputs = self.model.forward(G)  # Model forward pass

        # Get sentence nodes (dtype==1)
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        label = G.ndata["label"][snode_id].sum(-1)  # Node labels [n_nodes]

        # Compute loss per node
        G.nodes[snode_id].data["loss"] = self.criterion(outputs, label).unsqueeze(-1)
        loss = dgl.sum_nodes(G, "loss").mean()  # Batch mean
        self.running_loss += float(loss.data)

        G.nodes[snode_id].data["p"] = outputs  # Save predictions
        glist = dgl.unbatch(G)  # Split batch into individual graphs

        for j in range(len(glist)):
            idx = index[j]
            example = dataset.get_example(idx)
            original_article_sents = example.original_article_sents
            sent_max_number = len(original_article_sents)
            refer = example.original_abstract

            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            N = len(snode_id)
            p_sent = g.ndata["p"][snode_id].view(-1, 2)  # Predictions per node
            label = g.ndata["label"][snode_id].sum(-1).squeeze().cpu()

            # Decide which sentences to select
            if self.m == 0:
                prediction = p_sent.max(1)[1]
                pred_idx = torch.arange(N)[prediction != 0].long()
            else:
                if blocking:
                    pred_idx = self.ngram_blocking(
                        original_article_sents, p_sent[:, 1], self.blocking_win, min(self.m, N))
                else:
                    topk, pred_idx = torch.topk(p_sent[:, 1], min(self.m, N))
                prediction = torch.zeros(N).long()
                prediction[pred_idx] = 1

            self.extracts.append(pred_idx.tolist())

            # Update counters for evaluation metrics
            self.pred += prediction.sum()
            self.true += label.sum()
            self.match_true += ((prediction == label) & (prediction == 1)).sum()
            self.match += (prediction == label).sum()
            self.total_sentence_num += N
            self.example_num += 1

            # Build hypothesis text
            hyps = "\n".join(original_article_sents[id] for id in pred_idx if id < sent_max_number)
            self._hyps.append(hyps)
            self._refer.append(refer)

    def getMetric(self):
        """Calculate accuracy, precision, recall, and F-score after evaluation."""
        logger.info("[INFO] Validset match_true %d, pred %d, true %d, total %d, match %d",
                    self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        self._accu, self._precision, self._recall, self._F = eval_label(
            self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        logger.info(
            "[INFO] The size of totalset is %d, sent_number is %d, accu is %f, precision is %f, recall is %f, F is %f",
            self.example_num, self.total_sentence_num, self._accu, self._precision, self._recall, self._F)

    def ngram_blocking(self, sents, p_sent, n_win, k):
        """
        Apply n-gram blocking to prevent selecting overlapping sentences.

        Args:
            sents: List of sentences in the document
            p_sent: Tensor of sentence scores [num_sentences]
            n_win: N-gram size for blocking
            k: Max number of sentences to select

        Returns:
            LongTensor of selected sentence indices
        """
        ngram_list = []
        _, sorted_idx = p_sent.sort(descending=True)
        S = []

        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            overlap_flag = 0
            sent_ngram = []

            for i in range(len(pieces) - n_win):
                ngram = " ".join(pieces[i: (i + n_win)])
                if ngram in ngram_list:
                    overlap_flag = 1
                    break
                else:
                    sent_ngram.append(ngram)

            if overlap_flag == 0:
                S.append(idx)
                ngram_list.extend(sent_ngram)
                if len(S) >= k:
                    break

        return torch.LongTensor(S)

    @property
    def labelMetric(self):
        """Return the computed F-score after evaluation."""
        return self._F
