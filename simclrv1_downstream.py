# General imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
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
from ssl_study.simclrv1.downstream.data import GetDataloader
from ssl_study.simclrv1.downstream.models import (SimCLRv1DownStreamModel,
                                                  download_model)
from ssl_study.simclrv1.downstream.pipeline import SimCLRv1DownstreamPipeline

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string(
    "model_artifact_path", None, "Model checkpoint saved as W&B artifact."
)
flags.mark_flag_as_required("model_artifact_path")
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

    # Load the dataframes
    train_df = download_dataset("train", "labelled-dataset")
    valid_df = download_dataset("val", "labelled-dataset")

    # Preprocess the DataFrames
    train_paths, train_labels = preprocess_dataframe(train_df, is_labelled=True)
    valid_paths, valid_labels = preprocess_dataframe(valid_df, is_labelled=True)

    # Build dataloaders
    dataset = GetDataloader(config)
    trainloader = dataset.get_dataloader(
        train_paths, train_labels, dataloader_type="train"
    )
    validloader = dataset.get_dataloader(
        valid_paths, valid_labels, dataloader_type="valid"
    )

    # Download the model and load it.
    model_path = download_model(FLAGS.model_artifact_path)
    if wandb.run is not None:
        artifact = run.use_artifact(FLAGS.model_artifact_path, type="model")
    print("Path to the model checkpoint: ", model_path)

    model = tfsim.models.contrastive_model.load_model(model_path)

    # Build the model
    tf.keras.backend.clear_session()
    model = SimCLRv1DownStreamModel(config).get_model(model.backbone)
    model.summary()

    # Build the pipeline
    pipeline = SimCLRv1DownstreamPipeline(model, config, CALLBACKS)

    # Train and Evaluate
    pipeline.train_and_evaluate(train_paths, trainloader, validloader)


if __name__ == "__main__":
    app.run(main)
