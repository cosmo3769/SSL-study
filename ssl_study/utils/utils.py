import random
import string

import matplotlib.pyplot as plt
import numpy as np
import wandb


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def get_random_id():
    random_id = id_generator(size=8)
    configs.exp_id = random_id
    return configs.exp_id


# Sanity Check
class ShowBatch:
    """
    Script to show image batch in dataset

    Example usage:
        show_batch = ShowBatch(configs)
        sample_imgs, sample_labels = next(iter(validloader))
        show_batch.show_batch(sample_imgs, sample_labels)
    """

    def __init__(self, args):
        self.args = args

    def get_label(self, one_hot_label):
        label = np.argmax(one_hot_label, axis=0)
        return label

    def show_batch(self, image_batch, label_batch=None):
        plt.figure(figsize=(20, 20))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n])
            if label_batch is not None:
                plt.title(self.get_label(label_batch[n].numpy()))
            plt.axis("off")
