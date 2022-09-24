# General imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob

import numpy as np
import tensorflow as tf
from absl import app, flags
from ml_collections.config_flags import config_flags
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import LearningRateScheduler
from wandb.keras import WandbCallback

import wandb
# Import modules
from ssl_study.data import (GetDataloader, download_dataset,
                            preprocess_dataframe)
from ssl_study.models import SimpleSupervisedModel
from ssl_study.pipeline import SupervisedPipeline
from ssl_study import callbacks

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool(
    "log_eval", False, "Log model prediction, needs --wandb argument as well."
)


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    CALLBACKS = []
    sync_tensorboard = None
    if config.callback_config.use_tensorboard:
        sync_tensorboard = True
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            entity=CONFIG.value.wandb_config.entity,
            project=CONFIG.value.wandb_config.project,
            job_type="train",
            config=config.to_dict(),
            sync_tensorboard=sync_tensorboard
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [callbacks.WandBMetricsLogger()]


    # Load the dataframes
    train_df = download_dataset("train", "labelled-dataset")
    valid_df = download_dataset("val", "labelled-dataset")

    # Preprocess the DataFrames
    train_paths, train_labels = preprocess_dataframe(train_df, is_labelled=True)
    valid_paths, valid_labels = preprocess_dataframe(valid_df, is_labelled=True)

    # Compute class weights if use_class_weights is True.
    class_weights = None
    if config.bool_config.use_class_weights:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        class_weights = dict(zip(np.unique(train_labels), class_weights))

    # Build dataloaders
    dataset = GetDataloader(config)
    trainloader = dataset.get_dataloader(
        train_paths, train_labels, dataloader_type="train"
    )
    validloader = dataset.get_dataloader(
        valid_paths, valid_labels, dataloader_type="valid"
    )

    # Initialize callbacks
    callback_config = config.callback_config
    # Builtin early stopping callback
    if callback_config.use_earlystopping:
        earlystopper = callbacks.get_earlystopper(config)
        CALLBACKS += [earlystopper]
    # Built in callback to reduce learning rate on plateau
    if callback_config.use_reduce_lr_on_plateau:
        reduce_lr_on_plateau = callbacks.get_reduce_lr_on_plateau(config)
        CALLBACKS += [reduce_lr_on_plateau]

    # Initialize Model checkpointing callback
    if FLAGS.log_model:
        # Custom W&B model checkpoint callback
        model_checkpointer = callbacks.get_model_checkpoint_callback(config)
        CALLBACKS += [model_checkpointer]

    if wandb.run is not None:
        if FLAGS.log_eval:
            model_pred_viz = callbacks.get_evaluation_callback(
                config, validloader, DRIVABLE_SEG_MAP
            )
            CALLBACKS += [model_pred_viz]

    if callback_config.use_tensorboard:
        CALLBACKS += [tf.keras.callbacks.TensorBoard()]

    # Build the model
    tf.keras.backend.clear_session()
    model = SimpleSupervisedModel(config).get_model()
    model.summary()

    # Build the pipeline
    pipeline = SupervisedPipeline(model, config, class_weights, callbacks)

    # Train and Evaluate
    pipeline.train_and_evaluate(valid_df, trainloader, validloader)


if __name__ == "__main__":
    app.run(main)
