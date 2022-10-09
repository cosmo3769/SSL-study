import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_similarity as tfsim
import wandb
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import save_model
from tqdm import tqdm


class SimCLRv1Pipeline:
    def __init__(self, model, args, callbacks=[]):
        self.args = args
        self.model = model
        self.callbacks = callbacks

    def train_and_evaluate(self, inclass_paths, inclassloader):
        # Optimizer
        if self.args.train_config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(self.args.lr_config.init_lr_rate)
        elif self.args.train_config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                self.args.lr_config.init_lr_rate,
                self.args.train_config.sgd_momentum,
            )
        elif self.args.train_config.optimizer == "LAMB":
            optimizer = tfa.optimizers.LAMB(
                learning_rate=self.args.lr_config.init_lr_rate
            )
        else:
            raise NotImplementedError("This optimizer is not implemented.")

        # Loss
        loss = tfsim.losses.SimCLRLoss(
            name="simclr", temperature=self.args.train_config.temperature
        )

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss)

        # Train
        self.model.fit(
            inclassloader,
            epochs=self.args.train_config.epochs,
            steps_per_epoch=len(inclass_paths) // self.args.dataset_config.batch_size,
            # validation_data=val_ds,
            # validation_steps=VAL_STEPS_PER_EPOCH,
            # callbacks=[evb, tbc, mcp],
            callbacks=self.callbacks,
        )

        # # Evaluate
        # val_eval_loss, val_top_1_acc, val_top_5_acc = self.model.evaluate(validloader)

        # if wandb.run is not None:
        #     wandb.log(
        #         {
        #             "val_eval_loss": val_eval_loss,
        #             "val_top@1": val_top_1_acc,
        #             "val_top@5": val_top_5_acc,
        #         }
        #     )
