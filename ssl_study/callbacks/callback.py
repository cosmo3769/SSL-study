import tensorflow as tf
import wandb

class PolynomialDecay():
    def __init__(self, maxEpochs=5, init_lr_rate=0.001, power=1.0):
        self.maxEpochs = maxEpochs
        self.init_lr_rate = init_lr_rate
        self.power = power

    def __call__(self, epoch):
        decay = (1 - (epoch/float(self.maxEpochs))) ** self.power
        lr_rate = self.init_lr_rate * decay

        return float(lr_rate)

class GetCallbacks():
    def __init__(self, args):
        self.args = args

    def get_earlystopper(self):
        earlystopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.args.train_config.early_patience, verbose=0, mode='auto',
            restore_best_weights=True
        )

        return earlystopper

    def get_model_checkpoint(self):
        best_model = tf.keras.callbacks.ModelCheckpoint(
                                                        filepath = self.args.callback_config["filepath"],
                                                        monitor='val_top_1_acc',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='max',
                                                    )
        return best_model

    def get_reduce_lr_on_plateau(self):
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.args.train_config.rlrp_factor,
            patience=self.args.train_config.rlrp_patience
        )

        return reduce_lr_on_plateau