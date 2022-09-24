# General imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob

import numpy as np
import tensorflow as tf
from absl import app, flags
from callbacks import GetCallbacks, PolynomialDecay
# Import modules
from data import GetDataloader, download_dataset, preprocess_dataset
from ml_collections.config_flags import config_flags
from models import SimpleSupervisedModel
from pipeline import SupervisedPipeline
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import LearningRateScheduler
from wandb.keras import WandbCallback

import wandb
from configs.config import get_config

# Access all hyperparameter values through ml collection config
config = get_config()

# Initialize wandb
wandb.init(
    config=config.to_dict(),
    entity=config.wandb_config.entity,
    project=config.wandb_config.project,
)
# Access all hyperparameter values through wandb.config
config = wandb.config


def main():
    # Seed Everything
    tf.random.set_seed(config.seed)

    # Load the dataframes
    train_df = download_dataset("train", "labelled-dataset")
    valid_df = download_dataset("val", "labelled-dataset")

    # Preprocess the DataFrames
    train_paths, train_labels = preprocess_dataset(train_df)
    valid_paths, valid_labels = preprocess_dataset(valid_df)

    # Compute class weights if use_class_weights is True.
    class_weights = None
    if config.train_config["use_class_weights"]:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        class_weights = dict(zip(np.unique(train_labels), class_weights))

    # Build dataloaders
    dataset = GetDataloader(config)
    trainloader = dataset.dataloader(train_paths, train_labels, dataloader_type="train")
    validloader = dataset.dataloader(valid_paths, valid_labels, dataloader_type="valid")

    # Build the model
    tf.keras.backend.clear_session()
    model = SimpleSupervisedModel(config).get_model()
    model.summary()

    # Get learning rate schedulers
    if config.train_config["use_lr_scheduler"]:
        schedule = PolynomialDecay(
            maxEpochs=config.train_config["epochs"],
            init_lr_rate=config.lr_config["init_lr_rate"],
            power=5,
        )

    # Build callbacks
    # callback = GetCallbacks(config)
    callbacks = [WandbCallback(save_model=False), LearningRateScheduler(schedule)]

    # Build the pipeline
    pipeline = SupervisedPipeline(model, config, class_weights, callbacks)

    # Train and Evaluate
    pipeline.train_and_evaluate(valid_df, trainloader, validloader)


main()
