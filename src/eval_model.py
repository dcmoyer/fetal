

import os
import constants
import datetime
import glob
import numpy as np
import time
import util
from data_utils import DataGenerator
from models import UNet, UNetSmall, AESeg

from argparse import ArgumentParser

def eval_main(
  name,
  model_file,
  pred_list,
  input_location, output_location,
  organ="all_brains",
  input_file_format = "{s}/{s}_{n}.nii.gz",
  frame_reference = "",
  weights = None,
  resize=(None,None,None),
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
  input_file_format = f'{input_location}/{{s}}/{{s}}_{{n}}.nii.gz'

  #parse frame reference if string, otherwise leave as is
  if isinstance(frame_reference, str):
    if frame_reference == "all" or frame_reference == "all_frames":
      frame_reference = {key: range(length) for key, length in constants.SEQ_LENGTH.items() }
    elif frame_reference == "labeled_frames":
      frame_reference = new_labels.NEW_FRAMES

  shape = [i for i in resize] + [1] 
  model = MODELS["unet"](
    shape, name=name, filename=model_file, weights=weights,
  )

  #model.load_weights()

  pred_gen = DataGenerator( \
    {s: frame_reference[s] for s in pred_list}, \
    input_file_format, \
    load_files=False, \
    random_gen=False, \
    augment=False,
  )

  model.predict(pred_gen)


if __name__ == "__main__":
  ##
  ## parser
  ##

  parser = ArgumentParser()

  parser.add_argument('--name',
                      help='Name of model',
                      dest='name', type=str, required=True)

  parser.add_argument('--model',
                      help='Model architecture',
                      dest='model', type=str, required=True)

  parser.add_argument('--organ',
                      help='Organ to segment',
                      dest='organ', type=str, required=True)

  parser.add_argument('--model-file',
                      help='Pretrained model file',
                      dest='model_file', type=str)

  parser.add_argument('--data-prefix',
                      help='',
                      dest='data_prefix', type=str,
                      default='/data/vision/polina/projects/fetal_segmentation/data/rawdata/')

  parser.add_argument('--frame-ref',
                      help='',
                      dest='frame_ref', type=str,
                      default='labeled')

  # parser.add_argument('--split',
  #                     help='Train and validation split',
  #                     dest='split', type=float, nargs=2, default=[2/3, 1/6])
  # parser.add_argument('--load-files',
  #                     help='Load files',
  #                     dest='load_files', action='store_true')
  # parser.add_argument('--skip-training',
  #                     help='Skip training',
  #                     dest='skip_training', action='store_true')
  # parser.add_argument('--predict-all',
  #                     help='Predict all samples',
  #                     dest='predict_all', action='store_true')
  # parser.add_argument('--temporal',
  #                     help='Temporal segmentation using predictions from model name',
  #                     dest='temporal', type=str)
  # parser.add_argument('--good-frames',
  #                     help='Train using good frames from model name',
  #                     dest='good_frames', type=str)
  # parser.add_argument('--sample', type=str)

  options = parser.parse_args()

  eval_main(
    name=options.name,
    model=options.model,
    model_file=options.model_file,
    organ=options.organ,
    data_prefix=options.data_prefix,
    frame_reference=options.frame_ref,
  )

