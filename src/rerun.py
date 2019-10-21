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
from data_utils import DataGenerator
from models import UNet, UNetSmall, AESeg


#test, val, (implicit) train
options_split = [0.7,0.2]
options_epochs = 3000
options_load_files=True
organ = "all_brains"

##
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


np.random.seed(1919)

##
NEW_LABELED_SAMPLES = list( new_labels.NEW_LABELS.keys() )
n = len(NEW_LABELED_SAMPLES)
shuffled = np.random.permutation(NEW_LABELED_SAMPLES)

# set up split
train_split = int(options_split[0] * n)
val_split = int(np.sum(options_split) * n)

train = shuffled[:train_split]
val = shuffled[train_split:val_split]
test = shuffled[val_split:]

#frame_reference = constants.LABELED_FRAMES
frame_reference = new_labels.NEW_FRAMES
random_gen = False
shape = constants.SHAPE


#model_file = "/data/vision/polina/projects/fetal_segmentation/models/unet3000/2950_0.8471.h5"
#weights = util.get_weights(glob.glob(f'/data/vision/polina/projects/fetal_segmentation/data/labels/*/*_{organ}.nii.gz'))
name = "trial"
label_types = LABELS["unet"]
#model = MODELS["unet"](shape, name=name, filename=None, weights=weights)
model = MODELS["unet"](shape, name=name, filename=None, weights=None)


##
##
##
#input_file_format = '/data/vision/polina/projects/fetal_segmentation/data/rawdata/{s}/{s}_{n}.nii.gz'
input_file_format = 'data/vol-links-loc/{s}/{s}_{n}.nii.gz'
#input_file_format = 'data/downsampled_50/{s}/{s}_{n}.nii.gz'
label_file_format = f'data/label-links-loc/{{s}}/{{s}}_{{n}}_{organ}.nii.gz'
#label_file_format = f'/data/vision/polina/projects/fetal_segmentation/data/labels/{{s}}/{{s}}_{{n}}_{organ}.nii.gz'

#model.load_weights()

###
### train generator
###

train_gen = DataGenerator( \
  frames = {s: frame_reference[s] for s in train}, \
  input_file_format = input_file_format, \
  label_file_format = label_file_format, \
  label_types=label_types, \
  load_files=options_load_files, \
  random_gen=random_gen, \
  augment=True, \
)

val_gen = DataGenerator( \
  frames = {s: frame_reference[s] for s in val}, \
  input_file_format = input_file_format, \
  label_file_format = label_file_format, \
  label_types=label_types, \
  load_files=options_load_files, \
  random_gen=random_gen, \
  resize=True, \
)

logging.info(f'  Validation generator with {len(val_gen)} samples.')
logging.info('Training model.')

model.train(train_gen, val_gen, options_epochs)

exit(1)

###
###
###
pred_gen = DataGenerator( \
  {s: frame_reference[s] for s in test}, \
  input_file_format, \
  load_files=False, \
  random_gen=False, \
  tile_inputs=True, \
)

model.predict(pred_gen)

