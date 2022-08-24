from albumentations import augmentations
from albumentations.augmentations.transforms import ToGray
import numpy as np
import tensorflow as tf
from functools import partial
import albumentations as A

AUTOTUNE = tf.data.AUTOTUNE

class Augment():
    def __init__(self, args):
        self.args = args

    def build_augmentation(self, image):
        transform = A.Compose([
            A.RandomResizedCrop(self.args.augmentation_config["image_height"], 
                                self.args.augmentation_config["image_width"],
                                self.args.augmentation_config["cropscale"],
                                self.args.augmentation_config["cropratio"],
                                self.args.augmentation_config["probability"]),
            A.HorizontalFlip(self.args.augmentation_config["probability"]),
            A.ColorJitter(self.args.augmentation_config["jitterbrightness"], 
                          self.args.augmentation_config["jittercontrast"], 
                          self.args.augmentation_config["jittersaturation"], 
                          self.args.augmentation_config["jitterhue"], 
                          self.args.augmentation_config["alwaysapply"], 
                          self.args.augmentation_config["probability"]),
            A.ToGray(self.args.augmentation_config["probability"]),
            A.GaussianBlur(self.args.augmentation_config["gaussianblurlimit"], 
                           self.args.augmentation_config["gaussiansigmalimit"], 
                           self.args.augmentation_config["alwaysapply"], 
                           self.args.augmentation_config["probability"])
        ])
        return transform

    def augmentation(self, image):
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image], Tout=tf.float32)
        aug_img.set_shape((self.args.augmentation_config["image_height"], 
                           self.args.augmentation_config["image_width"], 3))

        aug_img = tf.image.resize(aug_img, 
                             [self.args.augmentation_config["image_height"], 
                             self.args.augmentation_config["image_width"]],
                             method='bicubic', 
                             preserve_aspect_ratio=False)
        aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
        
        return aug_img

    def aug_fn(self, image):
        data = {"image":image}
        aug_data = self.build_augmentation(**data)
        aug_img = aug_data["image"]

        return aug_img.astype(np.float32)