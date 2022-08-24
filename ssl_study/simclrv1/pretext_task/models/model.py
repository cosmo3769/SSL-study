import tensorflow as tf

class SimCLRv1Model():
    def __init__(self, args):
        self.args = args

    def get_backbone(self):
        """Get backbone for the model."""
        weights = None

        if self.args.model_config["backbone"] == 'resnet50':
            base_encoder = tf.keras.applications.ResNet50(include_top=False, weights=weights)
            base_encoder.trainabe = True
        else:
            raise NotImplementedError("Not implemented for this backbone.")

        return base_encoder

    def get_model(self, hidden_1, hidden_2, hidden_3):
        """Get model."""
        # Backbone
        base_encoder = self.get_backbone()

        # Stack layers
        inputs = tf.keras.layers.Input(
            (self.args.dataset_config["image_height"],
             self.args.dataset_config["image_width"],
             self.args.dataset_config["channels"]))

        x = base_encoder(inputs, training=True)

        projection_1 = tf.keras.layers.Dense(hidden_1)(x)
        projection_1 = tf.keras.layers.Activation("relu")(projection_1)
        projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
        projection_2 = tf.keras.layers.Activation("relu")(projection_2)
        projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

        return tf.keras.models.Model(inputs, projection_3)