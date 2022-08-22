import os
import ml_collections

def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "ssl-study"
    configs.entity = "wandb_fc"

    return configs

def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 224 #default - 224
    configs.image_width = 224 #default - 224
    configs.channels = 3
    configs.batch_size = 64
    configs.num_classes = 200

    return configs

def get_augment_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 224 #default - 224
    configs.image_width = 224 #default - 224
    configs.cropscale = (0.08, 1.0)
    configs.cropratio = (0.75, 1.3333333333333333)
    configs.jitterbrightness = 0.2
    configs.jittercontrast = 0.2
    configs.jittersaturation = 0.2
    configs.jitterhue = 0.2
    configs.gaussianblurlimit = (3, 7)
    configs.gaussiansigmalimit = 0
    configs.alwaysapply = False
    configs.probability = 0.5

    return configs

def get_bool_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.backbone = "resnet50"
    configs.apply_resize = True
    configs.do_cache = False
    configs.use_cosine_similarity = True

    return configs

def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 30
    configs.temperature = 0.5
    configs.s = 1 

    return configs
    
def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.augmentation_config = get_augment_configs()
    config.bool_config = get_bool_configs()
    config.train_config = get_train_configs()

    return config