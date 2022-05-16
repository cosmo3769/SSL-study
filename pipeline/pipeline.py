import os
import json
import tempfile
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
from utils.wandb_utils import id

class SimpleSupervisedModel():
    def __init__(self, args):
        self.args = args
        
    def get_resnet50(self):
        """
        Get baseline efficientnet model
        """
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        base_model.trainabe = True
        base_model_regularized = self.add_regularization(base_model, regularizer=tf.keras.regularizers.l2(0.0001))

        inputs = layers.Input((self.args.image_height, self.args.image_width, 3))
        x = base_model_regularized(inputs, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.args.num_labels, activation='softmax')(x)

        return models.Model(inputs, outputs)

    def add_regularization(self, model, regularizer=tf.keras.regularizers.l2(0.0001)):
        '''    
        Args:
            model: base_model(resnet50)
            regularizer: L2 regularization with value 0.0001
        
        Return: 
            model: base_model layers with regularization
        '''
        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
          print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
          return model

        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                  setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = tf.keras.models.model_from_json(model_json)
        
        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)
        return model

class SupervisedPipeline():
    def __init__(self, args):
        self.args = args

    def train(self):
        # Prepare Dataloaders
        dataset = GetDataloader(configs)
        trainloader = dataset.dataloader(train_df, data_type='train')

        # Prepare Model
        tf.keras.backend.clear_session()    
        get_model = SimpleSupervisedModel(configs)
        model = get_model.get_resnet50()

        # Compile model
        # optimizer = configs.optimizer(configs.learning_rate, configs.momentum)
        optimizer = tf.keras.optimizers.SGD(self.args.learning_rate, self.args.momentum)
        model.compile(optimizer,
                      loss=self.args.loss,
                      metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, name='top@1_acc'),
                              tf.keras.metrics.TopKCategoricalAccuracy(5, name='top@5_acc')])

        # Train
        model.fit(trainloader,
                  epochs=self.args.epochs,
                  validation_data=validloader,
                  callbacks=[WandbCallback(save_model=False)])

        return model

    def evaluate(self, model):
        # Prepare Dataloaders
        dataset = GetDataloader(configs)
        validloader = dataset.dataloader(valid_df, data_type='valid')

        # Evaluate
        val_eval_loss, val_top_1_acc, val_top_5_acc = model.evaluate(validloader)
        wandb.log({
            'val_eval_loss': val_eval_loss,
            'val_top@1': val_top_1_acc,
            'val_top@5': val_top_5_acc
        })

        return val_eval_loss, val_top_1_acc, val_top_5_acc

    def test(self, model):
        '''
        Test Prediction

        Return: 
            pred_max: predicted labels
            accuracy: accuracy score of predicted labels and ground truth labels
        '''
        dataset = GetDataloader(configs)
        testloader = dataset.dataloader(test_df, data_type='test')

        pred = model.predict(testloader)
        pred_max = np.argmax(pred, axis = 1)
        accuracy = accuracy_score(np.array(test_df['label']), np.argmax(pred, axis = 1))

        wandb.log({
            'accuracy_score': accuracy
        })

        return pred_max, accuracy

# Initialize W&B run
run = wandb.init(entity='wandb_fc',
                 project='ssl-study',
                 config=vars(configs),
                 group=f'{configs.exp_id}_baseline',
                 job_type='pipeline',
                 name=f'{configs.exp_id}_pipeline')

pipeline = SupervisedPipeline(configs)

# Train
train = pipeline.train()

# Evaluate
eval = pipeline.evaluate(train)

# Test
test = pipeline.test(train)

run.finish()