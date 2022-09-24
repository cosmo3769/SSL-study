from functools import partial

import albumentations as A
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class GetDataloader:
    def __init__(self, args):
        self.args = args

    def dataloader(self, paths, labels, dataloader_type="train"):
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

        # Shuffle if its for training
        if dataloader_type == "train":
            dataloader = dataloader.shuffle(self.args.dataset_config["batch_size"])

        # Load the image
        dataloader = dataloader.map(
            partial(self.parse_data, dataloader_type=dataloader_type),
            num_parallel_calls=AUTOTUNE,
        )

        if self.args.bool_config["do_cache"]:
            dataloader = dataloader.cache()

        # Add augmentation to dataloader for training
        if self.args.get("train_config", None):
            if (
                self.args.bool_config["use_augmentations"]
                and dataloader_type == "train"
            ):
                self.transform = self.build_augmentation()
                dataloader = dataloader.map(
                    self.augmentation, num_parallel_calls=AUTOTUNE
                )

        # Add general stuff
        dataloader = dataloader.batch(self.args.dataset_config["batch_size"]).prefetch(
            AUTOTUNE
        )

        return dataloader

    def decode_image(self, img, dataloader_type="train"):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Normalize image
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # resize the image to the desired size
        if self.args.bool_config["apply_resize"] and dataloader_type == "train":
            img = tf.image.resize(
                img,
                [
                    self.args.dataset_config["image_height"],
                    self.args.dataset_config["image_width"],
                ],
                method="bicubic",
                preserve_aspect_ratio=False,
            )
            img = tf.clip_by_value(img, 0.0, 1.0)
        elif self.args.bool_config["apply_resize"] and dataloader_type == "valid":
            img = tf.image.resize(
                img,
                [
                    self.args.train_config["model_img_height"],
                    self.args.train_config["model_img_width"],
                ],
                method="bicubic",
                preserve_aspect_ratio=False,
            )
            img = tf.clip_by_value(img, 0.0, 1.0)
        else:
            raise NotImplementedError("No data type")

        return img

    def parse_data(self, path, label, dataloader_type="train"):
        # Parse Image
        image = tf.io.read_file(path)
        image = self.decode_image(image, dataloader_type)

        # Parse Target
        label = tf.cast(label, dtype=tf.int64)
        if self.args.bool_config["apply_one_hot"]:
            label = tf.one_hot(label, depth=self.args.dataset_config["num_classes"])
        return image, label

    def build_augmentation(self):
        transform = A.Compose(
            [
                A.RandomResizedCrop(
                    self.args.augmentation_config["crop_height"],
                    self.args.augmentation_config["crop_width"],
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    p=0.8,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

        return transform

    def augmentation(self, image, label):
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image], Tout=tf.float32)
        aug_img.set_shape(
            (
                self.args.train_config["model_img_height"],
                self.args.train_config["model_img_width"],
                3,
            )
        )

        aug_img = tf.image.resize(
            aug_img,
            [
                self.args.train_config["model_img_height"],
                self.args.train_config["model_img_width"],
            ],
            method="bicubic",
            preserve_aspect_ratio=False,
        )
        aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)

        return aug_img, label

    def aug_fn(self, image):
        data = {"image": image}
        aug_data = self.transform(**data)
        aug_img = aug_data["image"]

        return aug_img.astype(np.float32)
