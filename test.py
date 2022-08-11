# General imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import wandb
from absl import app
from absl import flags
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from ml_collections.config_flags import config_flags
from tensorflow.keras.models import load_model

# Import modules
from ssl_study.data import download_dataset, preprocess_dataframe, GetDataloader
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
        test_paths, test_labels = preprocess_dataframe(test_df)

        # Build dataloader
        dataset = GetDataloader(config)
        testloader = dataset.dataloader(test_paths, test_labels, dataloader_type='test')

        # Load the model
        filepath = './saved_model'
        model = load_model(filepath, compile = True)

        # Generate predictions
        predictions = model.predict(testloader)
        
        # Test Accuracy
        test_accuracy = accuracy_score(np.array(test_df['label']), np.argmax(predictions, axis = 1))

        print(test_accuracy)
        
        # wandb log test accuracy
        if wandb.run is not None:
            wandb.log({
                'test_accuracy': test_accuracy
            })

if __name__ == "__main__":
    app.run(main)
