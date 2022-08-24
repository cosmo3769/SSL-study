import os
import json
import tempfile
import numpy as np
from sklearn.metrics import accuracy_score
import wandb
from tqdm import tqdm
import tensorflow as tf

from ssl_study.simclrv1.pretext_task.utils import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2, get_negative_mask
from ssl_study.simclrv1.pretext_task.data import Augment

class SimCLRv1Pipeline():
    def __init__(self, model, args):
        self.args = args
        self.model = model

    @tf.function
    def train_step(self, xis, xjs, model, optimizer, criterion, temperature):
        with tf.GradientTape() as tape:
            zis = model(xis)
            zjs = model(xjs)

            # normalize projection feature vectors
            zis = tf.math.l2_normalize(zis, axis=1)
            zjs = tf.math.l2_normalize(zjs, axis=1)

            l_pos = sim_func_dim1(zis, zjs)
            l_pos = tf.reshape(l_pos, (self.args.dataset_config["batch_size"], 1))
            l_pos /= temperature

            negatives = tf.concat([zjs, zis], axis=0)

            loss = 0

            for positives in [zis, zjs]:
                l_neg = sim_func_dim2(positives, negatives)

                labels = tf.zeros(self.args.dataset_config["batch_size"], dtype=tf.int32)

                l_neg = tf.boolean_mask(l_neg, get_negative_mask(self.args.dataset_config["batch_size"]))
                l_neg = tf.reshape(l_neg, (self.args.dataset_config["batch_size"], -1))
                l_neg /= temperature

                logits = tf.concat([l_pos, l_neg], axis=1) 
                loss += criterion(y_pred=logits, y_true=labels)

            loss = loss / (2 * self.args.dataset_config["batch_size"])

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    def train_simclr(self, model, dataset, optimizer, criterion, temperature=0.1, epochs=100):
        step_wise_loss = []
        epoch_wise_loss = []

        augment = Augment.augmentation()

        for epoch in tqdm(range(epochs)):
            for image_batch in dataset:
                a = augment(image_batch)
                b = augment(image_batch)

                loss = self.train_step(a, b, model, optimizer, criterion, temperature)
                step_wise_loss.append(loss)

            epoch_wise_loss.append(np.mean(step_wise_loss))
            wandb.log({"nt_xentloss": np.mean(step_wise_loss)})
            
            if epoch % 10 == 0:
                print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        return epoch_wise_loss, model