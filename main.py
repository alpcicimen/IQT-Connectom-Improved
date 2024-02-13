import os
import sys

module_path = os.path.abspath(os.path.join('./IQT'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.models import load_model

from scipy.ndimage import zoom

from tqdm import tqdm

from typing import List, Tuple

from IQT import *
import IQT.util

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

tf.config.set_visible_devices([], 'GPU')

if __name__ == '__main__':

    model: keras.Model = models.unet3d_t1(16, 16)
    model.load_weights(filepath="./output/UNet_NoT1/Run23")

    print("Hello World!")