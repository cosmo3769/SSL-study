import wandb
import string

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

random_id = id_generator(size=8)
configs.exp_id = random_id
print('Experiment Id: ', configs.exp_id)