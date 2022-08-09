# General imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import wandb
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.utils import class_weight
from ml_collections.config_flags import config_flags
from tensorflow.keras.callbacks import LearningRateScheduler

# Import modules
from ssl_study.data import download_dataset, preprocess_dataset, GetDataloader
from ssl_study.models import SimpleSupervisedModel
from ssl_study.callbacks import GetCallbacks, PolynomialDecay
from ssl_study.pipeline import SupervisedPipeline

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")

def main(_):
    with wandb.init(
        entity=CONFIG.value.wandb_config.entity,
        project=CONFIG.value.wandb_config.project,
        job_type='test',
        config=CONFIG.value.to_dict(),
    ):
        # Access all hyperparameter values through wandb.config
        config = wandb.config
        # Seed Everything
        tf.random.set_seed(config.seed)

        # Load the dataframe
        test_df = download_dataset('test', 'labelled-dataset')

        # Preprocess the DataFrame
        test_paths, test_labels = preprocess_dataset(test_df)

        # Build dataloader
        dataset = GetDataloader(config)
        testloader = dataset.dataloader(test_paths, test_labels, dataloader_type='test')

if __name__ == "__main__":
    app.run(main)