import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim

AUTOTUNE = tf.data.AUTOTUNE


class GetDataloader:
    def __init__(self, args):
        self.args = args

    def get_dataloader(self, paths):
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

        print(dataloader)

        # augmented view(2 views of same example in a batch)
        dataloader = dataloader.map(self.process, num_parallel_calls=AUTOTUNE)

        if self.args.bool_config["do_cache"]:
            dataloader = dataloader.cache()

        # Add general stuff
        dataloader = (
            dataloader.shuffle(self.args.dataset_config.shuffle_buffer)
            .batch(self.args.dataset_config.batch_size)
            .prefetch(AUTOTUNE)
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
                    self.args.dataset_config.image_height,
                    self.args.dataset_config.image_width,
                ],
                method="bicubic",
                preserve_aspect_ratio=False,
            )
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def img_scaling(self, img):
        return tf.keras.applications.imagenet_utils.preprocess_input(
            img, data_format=None, mode="torch"
        )

    def simclr_augmenter(self, img):
        """
        args:
            img: Single image tensor of shape (H, W, C)

        returns:
            A single image tensor of shape (H, W, C) with values between 0.0 and 1.0.
        """

        # random resize and crop. Increase the size before we crop.
        img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
            img,
            self.args.augmentation_config.image_height,
            self.args.augmentation_config.image_width,
            area_range=self.args.augmentation_config.crop_resize_area,
        )

        # The following transforms expect the data to be [0, 1]
        # img /= 255.

        # random color jitter
        def _jitter_transform(x):
            return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
                x,
                np.random.uniform(0.0, 0.4),
                np.random.uniform(0.0, 0.4),
                np.random.uniform(0.0, 0.4),
                np.random.uniform(0.0, 0.1),
                "multiplicative",
            )

        img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(
            _jitter_transform, p=self.args.augmentation_config.probability, x=img
        )

        # # random grayscale
        def _grascayle_transform(x):
            return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

        img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(
            _grascayle_transform, p=self.args.augmentation_config.probability, x=img
        )

        # optional random gaussian blur
        img = tfsim.augmenters.augmentation_utils.blur.random_blur(
            img,
            self.args.augmentation_config.image_height,
            self.args.augmentation_config.image_width,
            p=self.args.augmentation_config.probability,
        )

        # random horizontal flip
        img = tf.image.random_flip_left_right(img)

        # scale the data back to [0, 255]
        # img = img * 255.
        # img = tf.clip_by_value(img, 0., 255.)

        return img

    def process(self, img):
        view1 = self.simclr_augmenter(img)
        view1 = self.img_scaling(view1)
        view2 = self.simclr_augmenter(img)
        view2 = self.img_scaling(view2)

        return (view1, view2)
