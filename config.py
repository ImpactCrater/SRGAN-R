from easydict import EasyDict as edict
import json
from os.path import expanduser

config = edict()
config.TRAIN = edict()
config.VALID = edict()

# home path
config.home_path = expanduser("~")

# checkpoint location
config.checkpoint_path = config.home_path + '/SRGAN-R/checkpoint/'

# samples location
config.samples_path = config.home_path + '/SRGAN-R/samples/'

## Adam
config.TRAIN.sample_batch_size = 25
config.TRAIN.batch_size = 4
config.TRAIN.learning_rate = 1e-4

## adversarial learning
config.TRAIN.n_epoch_gan = 1000

## train set location
config.TRAIN.hr_img_path = config.home_path + '/SRGAN-R/HRImage_Training/'

## test set location
config.VALID.hr_img_path = config.home_path + '/SRGAN-R/HRImage_Validation/'
config.VALID.eval_img_path = config.home_path + '/SRGAN-R/LRImage_Evaluation/'
config.VALID.eval_img_name = '1.png'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
