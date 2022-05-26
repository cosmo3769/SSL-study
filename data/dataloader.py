import numpy as np
import tensorflow as tf
from functools import partial
import albumentations
from albumentations import Compose, RandomCrop, Resize

AUTOTUNE = tf.data.AUTOTUNE

class GetDataloader():
    def __init__(self, args):
        self.args = args

    def dataloader(self, paths, labels, dataloader_type='train'):
        '''
        Args:
            paths: List of strings, where each string is path to the image.
            labels: List of one hot encoded labels.
            dataloader_type: Anyone of one train, valid, or test

        Return:
            dataloader: train, validation or test dataloader
        '''
        # Consume dataframe
        dataloader = tf.data.Dataset.from_tensor_slices((paths, labels))

        # Load the image
        dataloader = (
            dataloader
            .map(partial(self.parse_data, dataloader_type=dataloader_type), num_parallel_calls=AUTOTUNE)
        )

        if self.args.dataset_config.do_cache:
            dataloader = dataloader.cache()

        # Shuffle if its for training
        if dataloader_type=='train':
            dataloader = dataloader.shuffle(self.args.dataset_config.batch_size)

        # Add augmentation to dataloader for training
        if self.args.train_config.use_augmentations and dataloader_type=='train':
            self.transform = self.build_augmentation(dataloader_type=dataloader_type)
            dataloader = dataloader.map(self.augmentation, num_parallel_calls=AUTOTUNE)

        # Add general stuff
        dataloader = (
            dataloader
            .batch(self.args.dataset_config.batch_size)
            .prefetch(AUTOTUNE)
        )

        return dataloader

    def decode_image(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Normalize image
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # resize the image to the desired size
        if self.args.dataset_config.apply_resize:
            img = tf.image.resize(img, 
                                  [self.args.dataset_config.image_height, self.args.dataset_config.image_width],
                                  method='bicubic', preserve_aspect_ratio=False)
            img = tf.clip_by_value(img, 0.0, 1.0)

        return img

    def parse_data(self, path, label, dataloader_type='train'):
        # Parse Image
        image = tf.io.read_file(path)
        image = self.decode_image(image)

        if dataloader_type in ['train', 'valid']:
            # Parse Target
            label = tf.cast(label, dtype=tf.int64)
            if self.args.dataset_config.apply_one_hot:
                label = tf.one_hot(
                    label,
                    depth=self.args.dataset_config.num_classes
                    )
            return image, label
        elif dataloader_type == 'test':
            return image
        else:
            raise NotImplementedError("Not implemented for this data_type")

    def build_augmentation(self, dataloader_type='train'):
        if dataloader_type=='train':
            transform = Compose([
                RandomCrop(90, 90, p=0.5),
                Resize(self.args.augmentation_config.crop_height, 
                       self.args.augmentation_config.crop_width, p=1),
            ])
        else:
            raise NotImplementedError("No augmentation")

        return transform
            
    def augmentation(self, image, label):
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image], Tout=tf.float32)
        aug_img.set_shape((self.args.augmentation_config.crop_height, 
                           self.args.augmentation_config.crop_width, 3))

        return aug_img, label

    def aug_fn(self, image):
        data = {"image":image}
        aug_data = self.transform(**data)
        aug_img = aug_data["image"]

        return aug_img.astype(np.float32)