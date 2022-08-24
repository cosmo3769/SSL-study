# General imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import wandb
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from ml_collections.config_flags import config_flags

# Import modules
from ssl_study.data import download_dataset, preprocess_dataframe
from ssl_study.simclrv1.pretext_task.data import GetDataloader
from ssl_study.simclrv1.pretext_task.models import SimCLRv1Model
from ssl_study.simclrv1.pretext_task.pipeline import SimCLRv1Pipeline

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")

def main(_):
    with wandb.init(
        entity=CONFIG.value.wandb_config.entity,
        project=CONFIG.value.wandb_config.project,
        job_type='simclrv1_pretext',
        config=CONFIG.value.to_dict(),
    ):
        # Access all hyperparameter values through wandb.config
        config = wandb.config
        # Seed Everything
        tf.random.set_seed(config.seed)

        # Load the dataframes
        inclass_df = download_dataset('in-class', 'unlabelled-dataset')

        # Preprocess the DataFrames
        inclass_paths = preprocess_dataframe(inclass_df, is_labelled=False)

        # Build dataloaders
        dataset = GetDataloader(config)
        inclassloader = dataset.dataloader(inclass_paths)

        # Model 
        tf.keras.backend.clear_session()
        model = SimCLRv1Model(config).get_model(config.model_config["hidden1"], config.model_config["hidden2"], config.model_config["hidden3"])
        model.summary()

if __name__ == "__main__":
    app.run(main)