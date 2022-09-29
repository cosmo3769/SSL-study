import os

import ml_collections


def randaugment_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.value_range = (0, 255)
    configs.augmentations_per_image = 3
    configs.magnitude = 0.5,
    configs.magnitude_stddev=0.15,
    configs.rate=0.9090909090909091,
    configs.geometric=True,

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.aug_seed = 1234
    config.randaugment = randaugment_config()

    return config
