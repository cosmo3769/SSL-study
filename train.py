# General imports
import os
import glob
import wandb
from absl import app
from absl import flags
import tensorflow as tf
from wandb.keras import WandbCallback
from ml_collections.config_flags import config_flags

# Import modules
from data import download_dataset, preprocess_dataset, GetDataloader
from models import SimpleSupervisedModel
from callbacks import GetCallbacks
from pipeline import SupervisedPipeline

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("configs")


def main(_):
    with wandb.init(
        entity=FLAGS.configs.wandb_config.entity,
        project=FLAGS.configs.wandb_config.project,
        job_type='train',
        config=FLAGS.configs.to_dict(),
    ):

        # Seed Everything
        tf.random.set_seed(FLAGS.configs.seed)

        # Load the dataframes
        train_df = download_dataset('train', 'labelled-dataset')
        valid_df = download_dataset('val', 'labelled-dataset')

        # Preprocess the DataFrames
        train_paths, train_labels = preprocess_dataset(train_df)
        valid_paths, valid_labels = preprocess_dataset(valid_df)

        # Build dataloaders
        dataset = GetDataloader(FLAGS.configs)
        trainloader = dataset.dataloader(train_paths, train_labels, dataloader_type='train')
        validloader = dataset.dataloader(valid_paths, valid_labels, dataloader_type='valid')

        # Build the model
        tf.keras.backend.clear_session()
        model = SimpleSupervisedModel(FLAGS.configs).get_model()
        model.summary()

        # Build callbacks
        callbacks = [WandbCallback()]

        # Build the pipeline
        pipeline = SupervisedPipeline(model, FLAGS.configs, callbacks)

        # Train and Evaluate
        pipeline.train_and_evaluate(trainloader, validloader)

if __name__ == "__main__":
    app.run(main)