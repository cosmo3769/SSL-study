import os
import json
import tempfile
import numpy as np
from sklearn.metrics import accuracy_score
import wandb
import tensorflow as tf


class SupervisedPipeline():
    def __init__(self, model, args, callbacks=[]):
        self.args = args
        self.model = model
        self.callbacks = callbacks

    def train_and_evaluate(self, trainloader, validloader):
        # Compile model
        if self.args.train_config.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(self.args.train_config.learning_rate)
        elif self.args.train_config.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(self.args.train_config.learning_rate, self.args.train_config.momentum)
        else:
            raise NotImplementedError("This optimizer is not implemented.")

        # Train
        self.model.compile(optimizer,
                           loss=self.args.train_config.loss,
                           metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, name='top@1_acc'),
                                    tf.keras.metrics.TopKCategoricalAccuracy(5, name='top@5_acc')])

        self.model.fit(trainloader,
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