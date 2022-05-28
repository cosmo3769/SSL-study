import os
import json
import tempfile
import numpy as np
from sklearn.metrics import accuracy_score
import wandb
import tensorflow as tf

class SupervisedPipeline():
    def __init__(self, model, args, class_weights=None, callbacks=[]):
        self.args = args
        self.model = model
        self.class_weights = class_weights
        self.callbacks = callbacks

    def train_and_evaluate(self, trainloader, validloader):
        decay_steps = 1000
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            self.args.train_config.learning_rate, decay_steps)
        lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(self.args.train_config.learning_rate,
                                                                     decay_steps=100000,
                                                                     decay_rate=0.96,
                                                                     staircase=True)
        learning_rate = lr_decayed_fn

        # Compile model
        if self.args.train_config.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif self.args.train_config.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate, self.args.train_config.momentum)
        else:
            raise NotImplementedError("This optimizer is not implemented.")

        # Train
        self.model.compile(optimizer,
                           loss=self.args.train_config.loss,
                           metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, name='top@1_acc'),
                                    tf.keras.metrics.TopKCategoricalAccuracy(5, name='top@5_acc')])

        self.model.fit(trainloader,
                       class_weight= self.class_weights,
                       epochs=self.args.train_config.epochs,
                       validation_data=validloader,
                       callbacks=self.callbacks)

        # Evaluate
        val_eval_loss, val_top_1_acc, val_top_5_acc = self.model.evaluate(validloader)
        if wandb.run is not None:
            wandb.log({
                'val_eval_loss': val_eval_loss,
                'val_top@1': val_top_1_acc,
                'val_top@5': val_top_5_acc
            })

    def test(self, testloader):
        '''Test Prediction'''
        pred = self.model.predict(testloader)
        pred_max = np.argmax(pred, axis = 1)
        # TODO: Fix this
        test_accuracy = accuracy_score(np.array(test_df['label']), np.argmax(pred, axis = 1))

        if wandb.run is not None:
            wandb.log({
                'test_accuracy': test_accuracy
            })