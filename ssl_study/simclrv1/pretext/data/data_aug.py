import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim

class Augment():
    def __init__(self, args):
        self.args = args

    def img_scaling(img):
      return tf.keras.applications.imagenet_utils.preprocess_input(
          img, 
          data_format=None, 
          mode='torch')


    @tf.function
    def simclr_augmenter(self, img, blur=True, area_range=(0.2, 1.0)):
        """
        args:
            img: Single image tensor of shape (H, W, C)
            blur: If true, apply blur. Should be disabled for cifar10.
            area_range: The upper and lower bound of the random crop percentage.

        returns:
            A single image tensor of shape (H, W, C) with values between 0.0 and 1.0.
        """

        # random resize and crop. Increase the size before we crop.
        img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
            img, self.args.augmentation_config.image_height, self.args.augmentation_config.image_width, area_range=self.args.augmentation_config.crop_resize_area
        )
        
        # The following transforms expect the data to be [0, 1]
        img /= 255.
        
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

        img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=self.args.augmentation_config.probability, x=img)

        # # random grayscale
        def _grascayle_transform(x):
            return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

        img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=self.args.augmentation_config.probability, x=img)

        # optional random gaussian blur
        img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, p=self.args.augmentation_config.probability)

        # random horizontal flip
        img = tf.image.random_flip_left_right(img)
        
        # scale the data back to [0, 255]
        img = img * 255.
        img = tf.clip_by_value(img, 0., 255.)

        return img


    @tf.function()
    def process(self, img):
        view1 = self.simclr_augmenter(img, blur=False)
        view1 = self.img_scaling(view1)
        view2 = self.simclr_augmenter(img, blur=False)
        view2 = self.img_scaling(view2)
        
        return (view1, view2)