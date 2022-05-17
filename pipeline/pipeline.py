import numpy as np
from sklearn.metrics import accuracy_score
import wandb
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras import regularizers

# Module imports
from configs.config import get_config
from dataloader.dataloader import GetDataloader
from utils.callback import callbacks
from utils.download_dataset import download_dataset

train_df = download_dataset('train', 'labelled-dataset')
valid_df = download_dataset('val', 'labelled-dataset')
test_df = download_dataset('test', 'labelled-dataset')

dataset = GetDataloader(get_config)
trainloader = dataset.dataloader(train_df, data_type='train')
validloader = dataset.dataloader(valid_df, data_type='valid')
testloader = dataset.dataloader(test_df, data_type='test')

class SupervisedPipeline():
    def __init__(self, model, args):
        self.args = args
        self.model = model

    def train_and_evaluate(self, trainloader):
        tf.keras.backend.clear_session()    

        # Compile model
        optimizer = tf.keras.optimizers.SGD(self.args.learning_rate, self.args.momentum)
        self.model.compile(optimizer,
                      loss=self.args.loss,
                      metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, name='top@1_acc'),
                              tf.keras.metrics.TopKCategoricalAccuracy(5, name='top@5_acc')])

        # Train
        self.model.fit(trainloader,
                  epochs=self.args.epochs,
                  validation_data=validloader,
                  callbacks=[WandbCallback(save_model=False)])

        # Evaluate
        val_eval_loss, val_top_1_acc, val_top_5_acc = self.model.evaluate(validloader)
        wandb.log({
            'val_eval_loss': val_eval_loss,
            'val_top@1': val_top_1_acc,
            'val_top@5': val_top_5_acc
        })

    def test(self, testloader):
        # Test
        pred = self.model.predict(testloader)
        pred_max = np.argmax(pred, axis = 1)
        accuracy = accuracy_score(np.array(test_df['label']), np.argmax(pred, axis = 1))

        wandb.log({
            'test_accuracy': accuracy
        })