import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE

class GetDataloader():
    def __init__(self, args):
        self.args = args
        
    def dataloader(self, df, data_type='train'):
        '''
        Args:
            df: Pandas dataframe
            data_type: Anyone of one train, valid, or test
            
        Return: 
            dataloader: train, validation or test dataloader
        '''
        # Consume dataframe
        dataloader = tf.data.Dataset.from_tensor_slices(dict(df))
        
        # Shuffle if its for training
        if data_type=='train':
            dataloader = dataloader.shuffle(self.args.batch_size)

        # Load the image
        dataloader = (
            dataloader
            .map(partial(self.parse_data, data_type=data_type), num_parallel_calls=AUTOTUNE)
            # .cache() # Comment if required
        )

        # Add general stuff
        dataloader = (
            dataloader
            .batch(self.args.batch_size)
            .prefetch(AUTOTUNE)
        )

        return dataloader

    @tf.function
    def decode_image(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Normalize image
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # resize the image to the desired size
        if self.args.resize:
            img = tf.image.resize(img, [self.args.image_height, self.args.image_width], 
                                  method='bicubic', preserve_aspect_ratio=False)
            img = tf.clip_by_value(img, 0.0, 1.0)

        return img

    @tf.function
    def parse_data(self, df_dict, data_type='train'):
        # Parse Image
        image = tf.io.read_file(df_dict['image_path'])
        image = self.decode_image(image)
        # image = tf.image.flip_left_right(image)

        if data_type in ['train', 'valid']:
            # Parse Target
            label = tf.cast(df_dict['label'], dtype=tf.int64)
            label = tf.one_hot(indices=label, depth=self.args.num_labels)
            return image, label
        elif data_type == 'test':
            return image
        else:
            raise NotImplementedError("Not implemented for this data_type")