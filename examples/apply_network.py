
import train_model
import eval_model 
import pandas as pd
import numpy as np

import pprint

##
##
##

##
##
##

n_inner_folds = 20

#if options.cv_idx > n_inner_folds or options.cv_idx < 0:
#  print("ERROR: Bad cv_idx, aborting.")
#  exit(1)

##
##
##

##
##
##

model_prefix = "larry_cv_test5"
weight_set_name = "2950_0.9208"

data_root = "/data/vision/polina/scratch/dmoyer/fetal"
input_location = "%s/data/vol-links-loc/" % data_root
label_location = "%s/data/label-links-loc/" % data_root
output_location = "%s/data/eval_%s_%s/" % (data_root, model_prefix, weight_set_name)

import os

os.makedirs(output_location, exist_ok=True)

import sep10_constants

list_of_subjects = ["MAP-C549"]

frame_ref = {subj : range(0,sep10_constants.SEQ_LENGTH[ subj ]) for subj in THE_ONE_SUBJ }

eval_model.eval_main(
  name="example",
  model_file="models/larry_cv_test5/2950_0.9208.h5",
  pred_list = THE_ONE_SUBJ,
  input_location=input_location, output_location=output_location,
  frame_reference = frame_ref,
)


