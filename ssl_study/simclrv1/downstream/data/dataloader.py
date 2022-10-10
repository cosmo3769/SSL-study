import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE


class GetDataloader:
    def __init__(self, args):
        self.args = args

    def get_dataloader(self, paths, labels, dataloader_type="train"):
        """
        Args:
            paths: List of strings, where each string is path to the image.
            labels: List of one hot encoded labels.
            dataloader_type: Anyone of one train, valid, or test

        Return:
            dataloader: train, validation or test dataloader
        """
        # Consume dataframe
        dataloader = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataloader = dataloader.repeat()

        # Shuffle if its for training
        if dataloader_type == "train":
            dataloader = dataloader.shuffle(self.args.dataset_config.shuffle_buffer)

        # Load the image
        dataloader = dataloader.map(
            partial(self.parse_data),
            num_parallel_calls=AUTOTUNE,
        )

        if self.args.dataset_config.do_cache:
            dataloader = dataloader.cache()

        # Add augmentation to dataloader for training
        if self.args.augmentation_config.use_augmentations and dataloader_type == "train":
            dataloader = dataloader.map(lambda x, y: (self.eval_augmenter(x), y), tf.data.AUTOTUNE)
            dataloader = dataloader.map(lambda x, y: (self.img_scaling(x), y), tf.data.AUTOTUNE)
        else: 
            dataloader = dataloader.map(lambda x, y: (self.img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)


        # Add general stuff
        dataloader = dataloader.batch(self.args.dataset_config.batch_size).prefetch(
            AUTOTUNE
        )

        return dataloader

    def parse_data(self, path, label):
        # Parse Image
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if self.args.bool_config["apply_resize"]:
            image = tf.image.resize(
                image,
                [
                    self.args.dataset_config.image_height,
                    self.args.dataset_config.image_width,
                ],
                method="bicubic",
                preserve_aspect_ratio=False,
            )
            image = tf.clip_by_value(image, 0.0, 1.0)

        label = tf.cast(label, dtype=tf.int64)
        if self.args.dataset_config.apply_one_hot:
            label = tf.one_hot(label, depth=self.args.dataset_config.num_classes)

        return image, label

    def img_scaling(self, img):
        return tf.keras.applications.imagenet_utils.preprocess_input(
            img, data_format=None, mode="torch"
        )

    def eval_augmenter(self, img):
        # random resize and crop. Increase the size before we crop.
        img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
            img,
            self.args.augmentation_config.image_height,
            self.args.augmentation_config.image_width,
            area_range=self.args.augmentation_config.crop_resize_area,
        )
        # random horizontal flip
        img = tf.image.random_flip_left_right(img)

        return img
