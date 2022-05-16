import numpy as np
import matplotlib.pyplot as plt

# Sanity Check
class ShowBatch():
    def __init__(self, args):
        self.args = args
        
    def get_label(self, one_hot_label):
        label = np.argmax(one_hot_label, axis=0)
        return label

    def show_batch(self, image_batch, label_batch=None):
        plt.figure(figsize=(20,20))
        for n in range(25):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            if label_batch is not None:
                plt.title(self.get_label(label_batch[n].numpy()))
            plt.axis('off')
            
show_batch = ShowBatch(configs)
sample_imgs, sample_labels = next(iter(validloader))
show_batch.show_batch(sample_imgs, sample_labels)