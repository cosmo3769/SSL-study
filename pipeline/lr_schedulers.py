import tensorflow as tf


class GetLRSchedulers():
    def __init__(self, args):
        self.args = args

    def get_cosine_decay(self):
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            self.args.lr_config.init_lr_rate, self.args.lr_config.cosine_decay_steps
        )
        return lr_scheduler

    def get_exponential_decay(self):
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            self.args.lr_config.init_lr_rate,
            decay_steps=self.args.lr_config.exp_decay_steps,
            decay_rate=self.args.lr_config.exp_decay_rate,
            staircase=self.args.lr_config.exp_is_staircase
        )
        return lr_scheduler