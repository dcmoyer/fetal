import os
import logging
logging.basicConfig(level=logging.INFO)
import new_labels

import constants
import datetime
import glob
import numpy as np
import time
import util
from data import DataGenerator
from models import UNet, UNetSmall, AESeg





##from argparse import ArgumentParser
##parser = ArgumentParser()
##parser.add_argument('--name',
##                    help='Name of model',
##                    dest='name', type=str, required=True)
##parser.add_argument('--model',
##                    help='Model architecture',
##                    dest='model', type=str, required=True)
##parser.add_argument('--organ',
##                    help='Organ to segment',
##                    dest='organ', type=str, required=True)
##parser.add_argument('--epochs',
##                    help='Training epochs',
##                    dest='epochs', type=int, default=1000)
##parser.add_argument('--split',
##                    help='Train and validation split',
##                    dest='split', type=float, nargs=2, default=[2/3, 1/6])
##parser.add_argument('--model-file',
##                    help='Pretrained model file',
##                    dest='model_file', type=str)
##parser.add_argument('--load-files',
##                    help='Load files',
##                    dest='load_files', action='store_true')
##parser.add_argument('--skip-training',
##                    help='Skip training',
##                    dest='skip_training', action='store_true')
##parser.add_argument('--predict-all',
##                    help='Predict all samples',
##                    dest='predict_all', action='store_true')
##parser.add_argument('--temporal',
##                    help='Temporal segmentation using predictions from model name',
##                    dest='temporal', type=str)
##parser.add_argument('--good-frames',
##                    help='Train using good frames from model name',
##                    dest='good_frames', type=str)
##parser.add_argument('--sample', type=str)
##options = parser.parse_args()

MODELS = {
    'unet': UNet,
    'unet-small': UNetSmall,
    'aeseg': AESeg,
}

LABELS = {
    'unet': ['label'],
    'unet-small': ['label'],
    'aeseg': ['label', 'input'],
}


#if options.temporal:
#    samples = list(constants.GOOD_FRAMES.keys())
#    n = len(samples)
#    shuffled = np.random.permutation(samples)
#    input_file_format = ['data/raw/{s}/{s}_{n}.nii.gz',
#                         'data/raw/{s}/{s}_{p}.nii.gz',
#                         f'data/predict_cleaned/{options.temporal}/{{s}}/{{s}}_{{p}}.nii.gz']
#    label_file_format = f'data/predict_cleaned/{options.temporal}/{{s}}/{{s}}_{{n}}.nii.gz'
#    random_gen = True
#    shape = constants.SHAPE[:-1] + (3,)
#elif options.good_frames:
#    samples = list(constants.GOOD_FRAMES.keys())
#    n = len(samples)
#    shuffled = np.random.permutation(samples)
#    input_file_format = 'data/raw/{s}/{s}_{n}.nii.gz'
#    label_file_format = f'data/predict_cleaned/{options.good_frames}/{{s}}/{{s}}_{{n}}.nii.gz'
#    random_gen = True
#    shape = constants.SHAPE
#else:

organ = "all_brains"


n = len(constants.LABELED_SAMPLES)
shuffled = np.random.permutation(constants.LABELED_SAMPLES)
#input_file_format = 'data/raw/{s}/{s}_{n}.nii.gz'
input_file_format = "/data/vision/polina/projects/fetal_segmentation/data/rawdata/{s}/{s}_{n}.nii.gz"
label_file_format = f'data/labels/{{s}}/{{s}}_{{n}}_{organ}.nii.gz'
random_gen = False
shape = constants.SHAPE

all_frames = {key: range(length) for key, length in constants.SEQ_LENGTH.items() }
frame_reference = all_frames
#frame_reference = new_labels.NEW_FRAMES

#model_file = "/data/vision/polina/projects/fetal_segmentation/models/unet3000/2950_0.8471.h5"
name = "no_ds"
model_file = "models/no_ds/0200_0.8704.h5"
#weights = util.get_weights(glob.glob(f'data/labels/*/*_{organ}.nii.gz'))

name = "ds50"
model_file = "models/ds50/0550_0.9071.h5"

model = MODELS["unet"](shape, name=name, filename=model_file, weights=None)

#model.load_weights()

pred_gen = DataGenerator( \
  {s: frame_reference[s] for s in shuffled}, \
  input_file_format, \
  load_files=False, \
  random_gen=False, \
  tile_inputs=True, \
)

model.predict(pred_gen)




