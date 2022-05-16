import wandb
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def id():
    random_id = id_generator(size=8)
    configs.exp_id = random_id
    return configs.exp_id