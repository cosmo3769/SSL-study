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
from ssl_study import callbacks
# Import modules
from ssl_study.data import (GetDataloader, download_dataset,
                            preprocess_dataframe)

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
            sync_tensorboard=sync_tensorboard,
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Load the Dataframes
    inclass_df = download_dataset('in-class', 'unlabelled-dataset')

    # Preprocess the DataFrames
    inclass_paths = preprocess_dataframe(inclass_df, is_labelled=False)

    # Build dataloaders
    dataset = GetDataloader(config)
    inclassloader = dataset.get_dataloader(inclass_paths)


if __name__ == "__main__":
    app.run(main)
