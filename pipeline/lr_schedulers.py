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