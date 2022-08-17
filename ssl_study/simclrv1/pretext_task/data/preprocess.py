import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

class PreprocessDataset():
    def __init__(self, args):
        self.args = args

    def preprocess_for_inclass(self):
        


    def parse_data(self, path, label, dataloader_type='train'):
        # Parse Image
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, 
                                  [self.args.augmentation_config["img_height"], 
                                   self.args.augmentation_config["img_width"]],
                                  method='bicubic', 
                                  preserve_aspect_ratio=False)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image