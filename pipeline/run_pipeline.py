import wandb

# Module imports
from utils.utils import get_random_id
from configs.config import get_config
from pipeline.pipeline import SupervisedPipeline

# Initialize W&B run
run = wandb.init(entity='wandb_fc',
                 project='ssl-study',
                 config=vars(configs),
                 group=f'{configs.exp_id}_baseline',
                 job_type='pipeline',
                 name=f'{configs.exp_id}_pipeline')

pipeline = SupervisedPipeline(get_config)

# Train and Evaluate
train = pipeline.train_and_evaluate()

# Test
test = pipeline.test(train)