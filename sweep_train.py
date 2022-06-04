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

# Import modules
from data import download_dataset, preprocess_dataset, GetDataloader
from models import SimpleSupervisedModel
from callbacks import GetCallbacks, CustomLearningRateScheduler
from pipeline import SupervisedPipeline, GetLRSchedulers
from configs.sweep_config import sweep_config

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("configs")


def main():
    # with wandb.init():
      with wandb.init(config=FLAGS.configs):
        config = wandb.config

        # FLAGS.configs.train_config.use_class_weights = config.use_class_weights
        # FLAGS.configs.train_

        # wandb.config.update = FLAGS.configs

        # Seed Everything
        tf.random.set_seed(FLAGS.configs.seed)

        # Load the dataframes
        train_df = download_dataset('train', 'labelled-dataset')
        valid_df = download_dataset('val', 'labelled-dataset')

        # Preprocess the DataFrames
        train_paths, train_labels = preprocess_dataset(train_df)
        valid_paths, valid_labels = preprocess_dataset(valid_df)

        # Compute class weights if use_class_weights is True.
        class_weights = None
        if FLAGS.configs.train_config.use_class_weights:
            class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                  classes=np.unique(train_labels), 
                                  y=train_labels)
            class_weights = dict(zip(np.unique(train_labels), class_weights))

        # Build dataloaders
        dataset = GetDataloader(FLAGS.configs)
        trainloader = dataset.dataloader(train_paths, train_labels, dataloader_type='train')
        validloader = dataset.dataloader(valid_paths, valid_labels, dataloader_type='valid')

        # Build the model
        tf.keras.backend.clear_session()
        model = SimpleSupervisedModel(FLAGS.configs).get_model()
        model.summary()

        # Get learning rate schedulers
        if FLAGS.configs.train_config.use_lr_scheduler:
            lr_schedulers = GetLRSchedulers(FLAGS.configs)

        # Build callbacks
        callback = GetCallbacks(FLAGS.configs)
        # callbacks = [WandbCallback(save_model=False), CustomLearningRateScheduler(lr_schedulers.get_exponential_decay())]
        callbacks = [WandbCallback()]
        
        # Build the pipeline
        pipeline = SupervisedPipeline(model, FLAGS.configs, class_weights, callbacks)

        # Train and Evaluate
        pipeline.train_and_evaluate(valid_df, trainloader, validloader)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, entity="wandb_fc", project="ssl-study")
    wandb.agent(sweep_id, main(), count = 10)