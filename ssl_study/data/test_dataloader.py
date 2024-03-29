import albumentations as A
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class GetTestDataloader:
    def __init__(self, args):
        self.args = args

    def dataloader(self, paths):
        """
        Args:
            paths: List of strings, where each string is path to the image.
        Return:
            dataloader: in-class dataloader
        """
        # Consume dataframe
        dataloader = tf.data.Dataset.from_tensor_slices(paths)

        # Load the image
        dataloader = dataloader.map(self.parse_data, num_parallel_calls=AUTOTUNE)

        if self.args.bool_config["do_cache"]:
            dataloader = dataloader.cache()

        # Add general stuff
        dataloader = dataloader.batch(self.args.dataset_config["batch_size"]).prefetch(
            AUTOTUNE
        )

        return dataloader

    def parse_data(self, path):
        # Parse Image
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.args.bool_config["apply_resize"]:
            image = tf.image.resize(
                image,
                [
                    self.args.test_config["model_img_height"],
                    self.args.test_config["model_img_width"],
                ],
                method="bicubic",
                preserve_aspect_ratio=False,
            )
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image
