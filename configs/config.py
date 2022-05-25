import os
import ml_collections


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "ssl-study"
    configs.entity = "wandb_fc"

    return configs

def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 224
    configs.image_width = 224
    configs.channels = 3
    configs.apply_resize = True
    configs.batch_size = 64
    configs.num_classes = 200
    configs.apply_one_hot = True
    configs.do_cache = True

    return configs

def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.backbone = "resnet50"
    configs.learning_rate = 0.0045
    configs.epochs = 10
    configs.use_pretrained_weights = True
    configs.regularize_backbone = True
    configs.l2_regularizer = 0.0001
    configs.post_gap_dropout = True
    configs.dropout_rate = 0.5
    configs.optimizer = "sgd"
    configs.momentum = 0.9
    configs.loss = "categorical_crossentropy"
    configs.early_patience = 6
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    configs.resume = False
    configs.reg = 0.0001

    return configs

def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.train_config = get_train_configs()

    return config