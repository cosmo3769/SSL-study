import tensorflow as tf
import tensorflow_similarity as tfsim


class SimCLRv1Model:
    def __init__(self, args):
        self.args = args

    def get_backbone(self, activation="relu", preproc_mode="torch"):
        input_shape = (
            self.args.dataset_config.image_height,
            self.args.dataset_config.image_width,
            self.args.dataset_config.channels,
        )

        if self.args.model_config.backbone == "resnet50":
            backbone = tfsim.architectures.ResNet50Sim(
                input_shape,
                include_top=False,  # Take the pooling layer as the output.
                pooling="avg",
            )
        else:
            raise NotImplementedError("Not implemented for this backbone.")

        return backbone

    def get_projector(self, input_dim, dim, activation="relu", num_layers: int = 3):
        inputs = tf.keras.layers.Input((input_dim,), name="projector_input")
        x = inputs

        for i in range(num_layers - 1):
            x = tf.keras.layers.Dense(
                dim,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.LecunUniform(),
                name=f"projector_layer_{i}",
            )(x)
            x = tf.keras.layers.BatchNormalization(
                epsilon=1.001e-5, name=f"batch_normalization_{i}"
            )(x)
            x = tf.keras.layers.Activation(
                activation, name=f"{activation}_activation_{i}"
            )(x)
        x = tf.keras.layers.Dense(
            dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name="projector_output",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            epsilon=1.001e-5,
            center=False,  # Page:5, Paragraph:2 of SimSiam paper
            scale=False,  # Page:5, Paragraph:2 of SimSiam paper
            name=f"batch_normalization_ouput",
        )(x)
        # Metric Logging layer. Monitors the std of the layer activations.
        # Degnerate solutions colapse to 0 while valid solutions will move
        # towards something like 0.0220. The actual number will depend on the layer size.
        o = tfsim.layers.ActivationStdLoggingLayer(name="proj_std")(x)

        projector = tf.keras.Model(inputs, o, name="projector")

        return projector
