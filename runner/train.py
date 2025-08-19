import os
import shutil
import time
import numpy as np
import torch
import dgl
from config import _DEBUG_FLAG_  # debug flag for logging
from data_manager import data_loaders  # module to create PyTorch/DGL dataloaders
from tools.logger import *  # logging utility
from tools.utils import save_model  # function to save PyTorch models
from runner.evaluation import run_eval  # evaluation function


def setup_training(model, hps, data_variables):
    """
    Prepares the training environment:
    - Restores a pretrained model if specified
    - Creates directories for saving models
    - Starts the training loop
    :param model: PyTorch model to train
    :param hps: hyperparameters object
    :param data_variables: dictionary containing dataset paths and vocab
    """
    train_dir = os.path.join(hps.save_root, "train")  # path to save training checkpoints

    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        # Load pre-trained model weights into model.HSG module
        model.HSG.load_state_dict(torch.load(hps.restore_model))
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)  # remove old training directory
        os.makedirs(train_dir)  # create new training directory

    try:
        run_training(model, hps, data_variables=data_variables)  # start training loop
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))  # save model on early stop


class Trainer:
    """
    Trainer class to handle training, batch processing, epoch management, 
    and saving model checkpoints.
    """

    def __init__(self, model, hps, train_dir):
        """
        Initializes training settings, optimizer, loss function, and tracking variables
        """
        self.model = model
        self.hps = hps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hps.lr)  # Adam optimizer
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')  # per-node loss
        self.best_train_loss = None  # best loss in training
        self.best_loss = None  # best validation loss
        self.best_F = None  # best validation F-score
        self.non_descent_cnt = 0  # early stop counter
        self.saveNo = 0  # checkpoint counter
        self.epoch = 1
        self.epoch_avg_loss = 0
        self.train_dir = train_dir
        self.report_epoch = 100  # how often to log batch info

    def run_epoch(self, train_loader):
        """
        Runs one epoch of training
        :param train_loader: DataLoader yielding batches
        :return: total loss for the epoch
        """
        epoch_start_time = time.time()  # start time of the epoch
        train_loss = 0.0  # cumulative loss for logging
        epoch_loss = 0.0  # cumulative loss for epoch

        iters_start_time = time.time()  # track batch times
        iter_start_time = time.time()

        for i, (G, index) in enumerate(train_loader):
            loss = self.train_batch(G=G)  # train one batch
            train_loss += float(loss.data)  # accumulate batch loss
            epoch_loss += float(loss.data)

            # Log batch statistics every report_epoch iterations
            if i % self.report_epoch == self.report_epoch - 1:
                if _DEBUG_FLAG_:  # if debug logging enabled
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())  # log gradient sum

                batch_time_sum = time.time() - iters_start_time
                iters_start_time = time.time()

                logger.info('| end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '.format(
                    i, (batch_time_sum / self.report_epoch), float(train_loss / self.report_epoch)))
                train_loss = 0.0
                self.save_current_model()  # save current checkpoint
            iter_start_time = time.time()

        # Compute average epoch loss
        self.epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info(' | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '.format(
            self.epoch, (time.time() - epoch_start_time), float(self.epoch_avg_loss)))
        return epoch_loss

    def train_batch(self, G):
        """
        Train model on a single batch of graphs
        :param G: DGL graph batch
        :return: batch loss
        """
        G = G.to(self.hps.device)  # move graph to GPU if necessary
        outputs = self.model.forward(G)  # forward pass: predictions [n_nodes, 2]

        # filter snodes (sentence nodes) and get their labels
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        label = G.ndata["label"][snode_id].sum(-1)  # aggregate multi-label

        # compute node-wise loss
        G.nodes[snode_id].data["loss"] = self.criterion(outputs, label.to(self.hps.device)).unsqueeze(-1)
        loss = dgl.sum_nodes(G, "loss")  # sum losses across nodes
        loss = loss.mean()  # mean over batch

        # check for numerical instability
        if not (np.isfinite(loss.data.cpu())).numpy():
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
            raise Exception("train Loss is not finite. Stopping.")

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        if self.hps.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hps.max_grad_norm)

        self.optimizer.step()  # update parameters
        return loss

    def change_learning_rate(self):
        """
        Adjusts learning rate using decay schedule
        """
        if self.hps.lr_descent:
            new_lr = max(5e-6, self.hps.lr / (self.epoch + 1))
            for param_group in list(self.optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

    def save_epoch_model(self):
        """
        Saves model if current epoch has best training loss
        """
        if not self.best_train_loss or self.epoch_avg_loss < self.best_train_loss:
            save_file = os.path.join(self.train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s',
                        float(self.epoch_avg_loss), save_file)
            save_model(self.model, save_file)
            self.best_train_loss = self.epoch_avg_loss
        elif self.epoch_avg_loss >= self.best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(self.model, os.path.join(self.train_dir, "earlystop"))
            sys.exit(1)

    def save_current_model(self):
        """
        Saves the current model state as 'current'
        """
        save_file = os.path.join(self.train_dir, "current")
        save_model(self.model, save_file)


def run_training(model, hps, data_variables):
    """
    Main training loop over multiple epochs and data partitions.
    - Splits large training data into parts to manage memory
    - Runs evaluation on validation set after each epoch
    - Handles early stopping
    """
    trainer = Trainer(model=model, hps=hps, train_dir=os.path.join(hps.save_root, "train"))
    train_size = 287000  # total number of training samples
    n_part = 16  # number of partitions to split training data

    for epoch in range(1, hps.n_epochs + 1):
        logger.info(f"train started in epoch={epoch}")
        trainer.epoch = epoch
        model.train()  # set model to training mode

        # Loop through partitions of training data
        for train_data_part in range(n_part + 1):
            if train_data_part == n_part:
                from_index = train_data_part * train_size // n_part
                to_index = None
            else:
                from_index = train_data_part * train_size // n_part
                to_index = (train_data_part + 1) * train_size // n_part

            # Create dataloader for this partition
            train_loader = data_loaders.make_dataloader(
                data_file=data_variables["train_file"],
                vocab=data_variables["vocab"], 
                hps=hps,
                filter_word=data_variables["filter_word"],
                w2s_path=data_variables["train_w2s_path"],
                graphs_dir=os.path.join(data_variables["graphs_dir"], "train"),
                from_index=from_index,
                to_index=to_index,
                shuffle=True
            )

            trainer.run_epoch(train_loader=train_loader)  # run epoch on this partition
            del train_loader  # free memory

        # Run validation
        valid_loader = data_loaders.make_dataloader(
            data_file=data_variables["valid_file"],
            vocab=data_variables["vocab"], 
            hps=hps,
            filter_word=data_variables["filter_word"],
            w2s_path=data_variables["val_w2s_path"],
            graphs_dir=os.path.join(data_variables["graphs_dir"], "val")
        )

        # Evaluate model on validation set
        best_loss, best_F, non_descent_cnt, saveNo = run_eval(
            model, valid_loader, valid_loader.dataset, hps,
            trainer.best_loss, trainer.best_F, trainer.non_descent_cnt,
            trainer.saveNo
        )

        del valid_loader  # free memory

        # Early stopping if validation loss does not improve for 3 epochs
        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(data_variables["train_dir"], "earlystop"))
            return
