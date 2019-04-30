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
from config import config

noise_level = config.TRAIN.noise_level

def rescale_m1p1(x):
    x = x / 127.5 - 1 # rescale to [－1, 1]
    return x

def get_imgs_fn(file_name, path):
    x = np.asarray(Image.open(path + file_name))
    return x

def save_img_fn(x, file_name):
    x = Image.fromarray(np.uint8(x))
    x.save(file_name)

def save_images(images, size, image_path='_temp.png'):
    """Save multiple images into one single image.

    Parameters
    -----------
    images : numpy array
        (batch, w, h, c)
    size : list of 2 ints
        row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : str
        save path

    """
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3), dtype=images.dtype)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, path):
        if np.max(images) <= 1 and (-1 <= np.min(images) < 0):
            images = ((images + 1) * 127.5).astype(np.uint8)
        elif np.max(images) <= 1 and np.min(images) >= 0:
            images = (images * 255).astype(np.uint8)

        return save_img_fn(merge(images, size), path)

    if len(images) > size[0] * size[1]:
        raise AssertionError("number of images should be equal or less than size[0] * size[1] {}".format(len(images)))

    return imsave(images, size, image_path)

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    return x

def downsample_fn(x):
    x = Image.fromarray(np.uint8(x)).resize((96, 96), Image.BICUBIC)
    q = random.randrange(noise_level, 101)
    img_file = BytesIO()
    x.save(img_file, 'webp', quality=q)
    x = Image.open(img_file)
    x = np.array(x) / 127.5 - 1
    return x
    
