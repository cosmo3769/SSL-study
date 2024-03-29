import os

import ml_collections


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "ssl-study"
    configs.entity = "wandb_fc"

    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 224  # default: 224
    configs.image_width = 224  # default: 224
    configs.channels = 3
    configs.shuffle_buffer = 1024
    configs.batch_size = 64
    configs.num_classes = 200
    configs.do_cache = False
    configs.use_augmentations = False
    # Always True since images are of varying sizes.
    configs.apply_resize = True
    configs.apply_one_hot = True

    return configs


def get_augmentation_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.crop_height = 224
    configs.crop_width = 224

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_height = 224
    configs.model_img_width = 224
    configs.model_img_channels = 3
    configs.backbone = "resnet50"
    configs.epochs = 1
    configs.l2_regularizer = 0.0001
    configs.dropout_rate = 0.5
    configs.optimizer = "sgd"
    configs.sgd_momentum = 0.9
    configs.loss = "categorical_crossentropy"
    configs.early_patience = 6
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    configs.reg = 0.0001

    return configs


def get_bool_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.use_pretrained_weights = True
    configs.regularize_backbone = True
    configs.use_class_weights = True
    configs.post_gap_dropout = True
    configs.resume = False

    return configs


def get_lr_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.init_lr_rate = 0.001
    configs.cosine_decay_steps = 1000
    configs.exp_decay_steps = 1000
    configs.exp_decay_rate = 0.96
    configs.exp_is_staircase = True

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
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


# TODO (ayulockin): remove get_config to a different py file
# and condition it with config_string as referenced here:
# https://github.com/google/ml_collections#parameterising-the-get_config-function
def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.augmentation_config = get_augmentation_configs()
    config.train_config = get_train_configs()
    config.lr_config = get_lr_configs()
    config.bool_config = get_bool_configs()
    config.callback_config = get_callback_configs()

    return config
