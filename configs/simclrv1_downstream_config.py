import os

import ml_collections


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "ssl-study"
    configs.entity = "wandb_fc"

    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 224  # default - 224
    configs.image_width = 224  # default - 224
    configs.channels = 3
    configs.batch_size = 64
    configs.num_classes = 200
    configs.shuffle_buffer = 1024
    configs.apply_one_hot = True
    configs.do_cache = False

    return configs


def get_augment_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 224  # default - 224
    configs.image_width = 224  # default - 224
    configs.crop_resize_area = (0.2, 1.0)
    configs.probability = 0.5
    configs.use_augmentations = True
    configs.alwaysapply = False

    return configs


def get_bool_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.apply_resize = True
    configs.do_cache = False

    return configs


def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.backbone = "resnet50"
    configs.projection_DIM = 2048
    configs.projection_layers = 2

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 50
    configs.temperature = 0.5
    configs.optimizer = "LAMB"

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    # Early stopping
    configs.use_earlystopping = True
    configs.early_patience = 6
    # Reduce LR on plateau
    configs.use_reduce_lr_on_plateau = False
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    # Model checkpointing
    configs.checkpoint_filepath = "wandb/model_{epoch}"
    configs.save_best_only = True
    # Model evaluation
    configs.viz_num_images = 100
    # Use tensorboard
    configs.use_tensorboard = False

    return configs


def get_lr_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.init_lr_rate = 1e-3

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.augmentation_config = get_augment_configs()
    config.bool_config = get_bool_configs()
    config.model_config = get_model_configs()
    config.train_config = get_train_configs()
    config.callback_config = get_callback_configs()
    config.lr_config = get_lr_configs()

    return config
