
import os
import logging
logging.basicConfig(level=logging.INFO)
import constants
import datetime
import glob
import numpy as np
import time
import util
#from data_utils import DataGenerator
from simple_data_utils import DataGenerator
from models import UNet, UNetSmall, AESeg

from argparse import ArgumentParser

def train_main(
  name,
  model_name,
  train_list, val_list,
  input_location,
  label_location,
  log_location,
  save_location,
  output_location,
  organ="all_brains",
  frame_reference = "",
  weights = None,
  options_epochs = 3000,
  options_augment = True,
  ):

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

  #TODO: fix this so organ is evaluated before setting this string
  input_file_format = f"{input_location}/{{s}}/{{s}}_{{n}}.nii.gz",
  label_file_format = f'{label_location}/{{s}}/{{s}}_{{n}}_{organ}.nii.gz'

  #parse frame reference if string, otherwise leave as is
  if isinstance(frame_reference, str):
    if frame_reference == "all" or frame_reference == "all_frames":
      frame_reference = {key: range(length) for key, length in constants.SEQ_LENGTH.items() }
    elif frame_reference == "labeled_frames":
      frame_reference = new_labels.NEW_FRAMES

  shape = constants.SHAPE
  model = MODELS[model_name]( \
    shape, name=name, weights=weights, \
    log_location=log_location, save_location=save_location, output_location=output_location \
  )

  #model.load_weights()

  # this seems...extra?
  label_types = LABELS[model_name]

  train_gen = DataGenerator( \
    list_of_samples = train_list, \
    frames={s: frame_reference[s] for s in train_list}, \
    input_file_format = input_file_format, \
    label_file_format = label_file_format, \
    label_types=label_types, \
    load_files=True, \
    random_gen=False, \
    augment=options_augment, \
  )

  val_gen = DataGenerator( \
    list_of_samples = val_list, \
    frames={s: frame_reference[s] for s in val_list}, \
    input_file_format = input_file_format, \
    label_file_format = label_file_format, \
    label_types=label_types, \
    load_files=True, \
    random_gen=False, \
    resize=True,\
    augment=False
  )

  logging.info(f'  Validation generator with {len(val_gen)} samples.')
  logging.info('Training model.')

  model.train(train_gen, val_gen, options_epochs)

  logging.info('Exiting training successfully.')
  return



