

import train_model
import pandas as pd
import numpy as np

import cv_util
import pprint

from argparse import ArgumentParser

##
##
##

parser = ArgumentParser()

parser.add_argument(
  '--cv-idx',
  dest='cv_idx', type=int,
  required=True
)
options = parser.parse_args()

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

df = pd.read_csv(
  "data/link_df.tsv",
  delimiter="\t",
  index_col=None,
  header=0,
  converters = {
    "l_frames" : lambda x : list(map(int,x.strip("[]").split(", "))),
    "id_idx" : int
  }
)

#df = df[ np.logical_not(df["is_twin"]) ]

df["cv_group"] = df["id_idx"] % n_inner_folds

tvt_splits, counts = cv_util.find_perms(
  n_uses=10,
  n_inner_folds=n_inner_folds,
  n_test_split=4,
  n_val_split=2,
  seed=1919
)

print(len(tvt_splits))

train, val, test = tvt_splits[ options.cv_idx ]
print(train, "\n\n" ,test, "\n\n" ,val)
test_set = df[df["cv_group"].isin(test)]
train_set = df[df["cv_group"].isin(train)]
val_set = df[df["cv_group"].isin(val)]

##
##
##

frame_ref = df[["id","l_frames"]].values.tolist()
frame_ref = {key : list(map(lambda x: int(x),val)) for key, val in frame_ref}

input_location = "data/vol-links-loc/"
label_location = "data/label-links-loc/"
#input_location = "/data/vision/polina/projects/fetal/data/BOLD/processed_data/split_nifti/mri/"
#label_location = "/data/vision/polina/projects/fetal/data/BOLD/processed_data/split_nifti/segmentation/brain_seg/"
#label_location = "/data/vision/polina/projects/fetal_segmentation/data/labels/"


train_list = train_set["id"].values.tolist()
val_list = val_set["id"].values.tolist()

train_model.train_main(
  name="cv_test%i" % options.cv_idx,
  model_name="unet",
  input_location=input_location, label_location=label_location,
  log_location="logs", save_location="models", output_location="data/predict/",
  train_list = train_list,
  val_list = val_list,
  frame_reference = frame_ref,
)



