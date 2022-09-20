import os
import json
import tempfile
import numpy as np
from sklearn.metrics import accuracy_score
import wandb
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import save_model

class SupervisedPipeline():
    def __init__(self, model, args, class_weights=None, callbacks=[]):
        self.args = args
        self.model = model
        self.class_weights = class_weights
        self.callbacks = callbacks

    def train_and_evaluate(self, valid_df, trainloader, validloader):
        # Compile model
        if self.args.train_config["optimizer"] == 'adam':
            optimizer = tf.keras.optimizers.Adam(self.args.lr_config["init_lr_rate"])
        elif self.args.train_config["optimizer"] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(self.args.lr_config["init_lr_rate"], self.args.train_config["sgd_momentum"])
        else:
            raise NotImplementedError("This optimizer is not implemented.")

        # Train
        self.model.compile(optimizer,
                           loss=self.args.train_config["loss"],
                           metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, name='top@1_acc'),
                                    tf.keras.metrics.TopKCategoricalAccuracy(5, name='top@5_acc')])

        self.model.fit(trainloader,
                       class_weight= self.class_weights,
                       epochs=self.args.train_config["epochs"],
                       validation_data=validloader,
                       callbacks=self.callbacks)

        # Evaluate
        val_eval_loss, val_top_1_acc, val_top_5_acc = self.model.evaluate(validloader)

        if self.args.bool_config["use_log_validation_table"]:
          validation_table = wandb.Table(columns=["image_id", "image", "true_labels", "evaluated_labels"])
          evaluation = self.model.predict(validloader)
          for i, tmp_df in tqdm(valid_df.iterrows()):
              validation_table.add_data(
                  int(tmp_df.image_id),
                  wandb.Image(tmp_df.image_path),
                  int(tmp_df.label),
                  int(np.argmax(evaluation[i], axis = 0))
              )

        if wandb.run is not None:
          if self.args.bool_config["use_log_validation_table"]:
            wandb.log({
                'val_eval_loss': val_eval_loss,
                'val_top@1': val_top_1_acc,
                'val_top@5': val_top_5_acc,
                'val_table': validation_table
            })
          else:
            wandb.log({
                'val_eval_loss': val_eval_loss,
                'val_top@1': val_top_1_acc,
                'val_top@5': val_top_5_acc
            })