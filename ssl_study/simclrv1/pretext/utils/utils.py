import numpy as np
import tensorflow_similarity as tfsim

# Sanity Check
class ShowAugmentedBatch:
    """
    Script to show image batch in dataset
    Example usage:
        show_augmented_batch = ShowAugmentedBatch(config)
        display_imgs = next(inclassloader.as_numpy_iterator())
        max_pixel = np.max([display_imgs[0].max(), display_imgs[1].max()])
        min_pixel = np.min([display_imgs[0].min(), display_imgs[1].min()])
        show_augmented_batch.show_augmented_batch(display_imgs, max_pixel, min_pixel)
    """

    def __init__(self, args):
        self.args = args

    def show_augmented_batch(self, display_imgs, max_pixel, min_pixel):
        tfsim.visualization.visualize_views(
            views=display_imgs,
            num_imgs=16,
            views_per_col=8,
            max_pixel_value=max_pixel,
            min_pixel_value=min_pixel,
        )



