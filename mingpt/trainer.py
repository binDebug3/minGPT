"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
import numpy as np

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()

        # CHANGE add checkpoint name
        checkpoint_name = config.checkpoint_name if hasattr(config, 'checkpoint_name') else 'checkpoint'
        # CHANGE set iternum to model if model has iter_num attribute
        self.iter_num = model.iter_num if hasattr(model, 'iter_num') else 0
        # CHANGE update last save
        self.since_last_save = 0
        # CHANGE update checkpoint number
        self.checkpoint_num = model.checkpoint_num if hasattr(model, 'checkpoint_num') else 0 
        # CHANGE save loss
        self.saved_loss = model.saved_loss if hasattr(model, 'saved_loss') else []
        # CHANGE set loss
        self.loss = self.saved_loss[-1] if self.saved_loss else np.inf
        self.current_loss = []

        # CHANGE add last save time
        self.since_last_save = 0
        self.iter_time = time.time()
        # CHANGE add iteration list
        self.iter_list = model.iter_list if hasattr(model, 'iter_list') else []
        data_iter = iter(train_loader)

        #
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # CHANGE squeeze input and output
            x = x.squeeze()
            y = y.squeeze()

            # forward the model
            # CHANGE replace model loss with initialize loss
            # logits, self.loss = model(x, y)
            prev_loss = self.loss
            logits, self.loss = model(x, y)
            self.current_loss.append(self.loss.detach())

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # CHANGE update save count
            self.since_last_save += 1

            # CHANGE implement checkpoint save
            if self.loss <= prev_loss and self.since_last_save >= config.checkpoint_iters:
                self.since_last_save = 0
                self.checkpoint_num += 1

                # create checkpoint dictionary
                checkpoint = { 
                    'model_transformer': model.transformer.state_dict(),
                    'model_lm_head': model.lm_head.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                    'iter_num': self.iter_num,
                    'iter_list': self.iter_list.append(self.iter_num),
                    'checkpoint_num': self.checkpoint_num, 
                    'saved_loss': self.saved_loss.append(self.loss),
                }

                # save checkpoint
                torch.save(checkpoint, 'checkpoints/checkpoint_{}.pth'.format(self.checkpoint_num))

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
