import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
from PIL import Image
import random
from io import BytesIO

def rescale_m1p1(x):
    x = x / 127.5 - 1 # rescale to [－1, 1]
    return x

def get_imgs_fn(file_name, path):
    x = np.asarray(Image.open(path + file_name))
    return x

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    return x

def downsample_fn(x):
    x = Image.fromarray(np.uint8(x)).resize((96, 96), Image.BICUBIC)
    q = random.randrange(50, 101)
    img_file = BytesIO()
    x.save(img_file, 'webp', quality=q)
    x = Image.open(img_file)
    x = np.array(x) / 127.5 - 1
    return x
    
