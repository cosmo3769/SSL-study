import json
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

import wandb


def regularize_backbone(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    """
    Args:
        model: base_model(resnet50)
        regularizer: L2 regularization with value 0.0001

    Return:
        model: base_model layers with regularization
    """
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ["kernel_regularizer"]:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp_weights.h5")
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


class SimpleSupervisedModel:
    def __init__(self, args):
        self.args = args

    def get_backbone(self):
        """Get backbone for the model."""
        weights = None
        if self.args.bool_config["use_pretrained_weights"]:
            weights = "imagenet"

        if self.args.train_config["backbone"] == "resnet50":
            base_model = tf.keras.applications.ResNet50(
                include_top=False, weights=weights
            )
            base_model.trainabe = True
            if self.args.bool_config["regularize_backbone"]:
                base_model = regularize_backbone(
                    base_model,
                    regularizer=tf.keras.regularizers.l2(
                        self.args.train_config["l2_regularizer"]
                    ),
                )
        else:
            raise NotImplementedError("Not implemented for this backbone.")

        return base_model

    def get_model(self):
        """Get model."""
        # Backbone
        base_model = self.get_backbone()

        # Stack layers
        inputs = tf.keras.layers.Input(
            (
                self.args.train_config["model_img_height"],
                self.args.train_config["model_img_width"],
                self.args.train_config["model_img_channels"],
            )
        )

        x = base_model(inputs, training=True)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if self.args.bool_config["post_gap_dropout"]:
            x = tf.keras.layers.Dropout(self.args.train_config["dropout_rate"])(x)
        outputs = tf.keras.layers.Dense(
            self.args.dataset_config["num_classes"], activation="softmax"
        )(x)

        return tf.keras.models.Model(inputs, outputs)
