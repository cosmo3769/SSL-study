import os
import json
import tempfile

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras import regularizers

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