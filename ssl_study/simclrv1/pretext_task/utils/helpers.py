import tensorflow as tf
import numpy as np

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

def get_criterion():
   return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

class Optimizer():
    def __init__(self, args):
        self.args = args

    def get_optimizer(self): 
        learning_rate = tf.keras.experimental.CosineDecay(initial_learning_rate= self.args.learning_rate_config['initial_learning_rate'], decay_steps=self.args.learning_rate_config['decay_steps'])
        optimizer = tf.keras.optimizers.SGD(learning_rate)

        return optimizer