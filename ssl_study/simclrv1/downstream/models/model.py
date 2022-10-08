import tensorflow as tf
import tensorflow_similarity as tfsim


class SimCLRv1DownStreamModel:
    def __init__(self, args):
        self.args = args

    def get_model(self, contrastive_model):
        input_shape = (
            self.args.dataset_config.image_height,
            self.args.dataset_config.image_width,
            self.args.dataset_config.channels,
        )

        inputs = tf.keras.layers.Input(input_shape, name="eval_input")
        x = contrastive_model(inputs, training=True)
        outputs = tf.keras.layers.Dense(200, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        return model
