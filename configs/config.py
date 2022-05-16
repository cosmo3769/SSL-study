from argparse import Namespace

configs = Namespace(
    # SEED
    seed = 42,
    image_height = 224,
    image_width = 224,
    resize=True,

    # CROSS VALIDATION
    num_folds = -1,

    # TRAIN
    batch_size = 64,
    epochs = 20, # default 20
    early_patience = 6,
    rlrp_factor = 0.2,
    rlrp_patience = 3,
    learning_rate = 0.0045,
    momentum = 0.9,
    resume = False,
    optimizer = 'SGD',
    loss = 'categorical_crossentropy',

    # MODEL
    model_save_path = 'models',
    model_type = 'resnet50', # 'resnet50'
)

configs.num_labels = len(train_df.label.unique())
print('Number of labels: ', configs.num_labels)

def get_config():
    return configs