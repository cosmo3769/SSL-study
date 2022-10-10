# General imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob

import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim
import wandb
from absl import app, flags
from ml_collections.config_flags import config_flags
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import LearningRateScheduler
from wandb.keras import WandbCallback

from ssl_study import callbacks
# Import modules
from ssl_study.data import download_dataset, preprocess_dataframe
from ssl_study.simclrv1.pretext.data import GetDataloader
from ssl_study.simclrv1.pretext.models import SimCLRv1Model
from ssl_study.simclrv1.pretext.pipeline import SimCLRv1Pipeline

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
            sync_tensorboard=sync_tensorboard,
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Load the Dataframes
    inclass_df = download_dataset("in-class", "unlabelled-dataset")

    # Preprocess the DataFrames
    inclass_paths = preprocess_dataframe(inclass_df, is_labelled=False)
    train_path = inclass_paths[2000:]
    val_path = inclass_paths[0:2000]

    # Build dataloaders
    dataset = GetDataloader(config)
    inclasstrainloader = dataset.get_dataloader(train_path)
    inclassvalloader = dataset.get_dataloader(val_path)

    # Model
    tf.keras.backend.clear_session()
    backbone = SimCLRv1Model(config).get_backbone()
    # backbone.summary()
    projector = SimCLRv1Model(config).get_projector(
        input_dim=backbone.output.shape[-1],
        dim=config.model_config.projection_DIM,
        num_layers=config.model_config.projection_layers,
    )
    # projector.summary()
    contrastive_model = tfsim.models.ContrastiveModel(
        backbone=backbone,
        projector=projector,
        algorithm="simclr",
    )
    contrastive_model.summary()

    # Initialize Model checkpointing callback
    callback_config = config.callback_config
    if FLAGS.log_model:
        # Custom W&B model checkpoint callback
        model_checkpointer = callbacks.get_model_checkpoint_callback(config)
        CALLBACKS += [model_checkpointer]

    if callback_config.use_tensorboard:
        CALLBACKS += [tf.keras.callbacks.TensorBoard()]

    # Build the pipeline
    pipeline = SimCLRv1Pipeline(contrastive_model, config, CALLBACKS)

    # Train and Evaluate
    pipeline.train_and_evaluate(inclass_paths, inclasstrainloader, inclassvalloader)


if __name__ == "__main__":
    app.run(main)
