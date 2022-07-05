import tensorflow as tf
import wandb

from tensorflow.keras.optimizers.schedules import ExponentialDecay

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_train_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        if isinstance(self.schedule, ExponentialDecay):
            scheduled_lr = self.schedule(batch)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        # Log the LR to W&B.
        print(batch, lr, scheduled_lr)
        if wandb.run is not None:
            wandb.log({'lr': scheduled_lr})

class GetCallbacks():
    def __init__(self, args):
        self.args = args

    def get_earlystopper(self):
        earlystopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.args.train_config.early_patience, verbose=0, mode='auto',
            restore_best_weights=True
        )

        return earlystopper

    def get_reduce_lr_on_plateau(self):
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.args.train_config.rlrp_factor,
            patience=self.args.train_config.rlrp_patience
        )

        return reduce_lr_on_plateau

    def get_lr_logger(self):
        lr_logger = LRLogger()
        return lr_logger