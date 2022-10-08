import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class SimCLRv1DownstreamPipeline:
    def __init__(self, model, args, class_weights=None, callbacks=[]):
        self.args = args
        self.model = model
        self.callbacks = callbacks

    def train_and_evaluate(self, train_paths, trainloader, validloader):
        # Optimizer
        if self.args.train_config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(self.args.lr_config.init_lr_rate)
        elif self.args.train_config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                self.args.lr_config.init_lr_rate,
                self.args.train_config.sgd_momentum,
            )
        else:
            raise NotImplementedError("This optimizer is not implemented.")

        # Compile
        self.model.compile(
            optimizer,
            loss=self.args.train_config.loss,
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(1, name="top@1_acc"),
                tf.keras.metrics.TopKCategoricalAccuracy(5, name="top@5_acc"),
            ],
        )

        # Train
        self.model.fit(
            trainloader,
            batch_size=self.args.dataset_config.batch_size,
            epochs=self.args.train_config.epochs,
            steps_per_epoch=int(len(train_paths) // self.args.dataset_config.batch_size),
            validation_data=validloader,
            validation_steps=self.args.train_config.val_steps_per_epoch
        )

        # Evaluate
        val_eval_loss, val_top_1_acc, val_top_5_acc = self.model.evaluate(validloader)

        if wandb.run is not None:
            wandb.log(
                {
                    "val_eval_loss": val_eval_loss,
                    "val_top@1": val_top_1_acc,
                    "val_top@5": val_top_5_acc,
                }
            )
