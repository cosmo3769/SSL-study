import os
import ml_collections


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "ssl-study"
    configs.entity = "wandb_fc"

    return configs

def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.batch_size = 64
    configs.num_classes = 200

    return configs

def get_test_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_height = 224
    configs.model_img_width = 224
    configs.model_img_channels = 3
    configs.backbone = "resnet50"

    return configs

def get_bool_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.apply_resize = True
    configs.do_cache = False

    return configs

def get_modelcheckpoint_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.filepath = './best-model'

    return configs

# TODO (ayulockin): remove get_config to a different py file
# and condition it with config_string as referenced here:
# https://github.com/google/ml_collections#parameterising-the-get_config-function
def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.test_config = get_test_configs()
    config.bool_config = get_bool_configs()
    config.modelcheckpoint_config = get_modelcheckpoint_configs()

    return config