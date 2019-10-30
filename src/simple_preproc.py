

import numpy as np
from util import read_vol

from collections import defaultdict

#
# 191021 
#
# This function takes the volume to the correct shape no matter what, zero padding or cropping
# to get there. 
#
# He's a very aggressive function, sorry, but he makes nice batches once you get to know him.
#
# zeropadding comes first! then cropping.
def resize_zero_padding_nd(vol, shape):

  n_dims = len(vol.shape)

  dxyz = []
  dxyz_mod = []
  for i in range(n_dims):
    if i < len(shape) and shape[i] is not None:
      dxyz.append((shape[i] - vol.shape[i]) // 2)
      dxyz_mod.append( (shape[i] - vol.shape[i]) % 2 )
    else:
      dxyz.append(None)
      dxyz_mod.append(None)

  crop_limits = []
  zpad_limits = []
  for i,(diff,diff_mod) in enumerate(zip(dxyz, dxyz_mod)):

    if i < len(shape) and diff is not None and diff > 0:
      # zero pad
      zpad_limits.append( (diff, shape[i] - vol.shape[i] + diff_mod) )
      crop_limits.append( (0,shape[i]) )
    elif i < len(shape) and diff is not None and diff < 0:
      # crop
      zpad_limits.append( (0,0) )
      crop_limits.append( (-diff,vol.shape[i] + diff + diff_mod) )
      
    else:
      # no change, either None or diff == 0
      zpad_limits.append( (0,0) )
      crop_limits.append( (0,vol.shape[i]) )

  resized = np.pad(vol, tuple(zpad_limits), 'constant')
  slices = tuple( slice(int(start),int(stop)) for start,stop in crop_limits )
  resized = resized[ slices ]

  return resized

#
# 191029
#
# this function takes one volume and turns it into maximally 8, where the
# original volume is tiled by the new ones. If the original shape is 150% of
# the new ones, an error will be thrown. This can be bypassed with the
# bypass_vol_error flag.
#
# only for 3d volumes
def get_subvolumes(vol, shape, bypass_vol_error=False):
  if np.all([vol.shape[i] <= shape[i] for i in range(3)]):
    return [ resize_zero_padding_nd(vol, shape) ],[ (0,0,0) ]

  slices = defaultdict(lambda : [])
  for i in range(3):
    slices[i].append(slice(0,np.min((vol.shape[i],shape[i]))))
    if vol.shape[i] > shape[i]:
      slices[i].append(slice(vol.shape[i] - shape[i], vol.shape[i]))
    if vol.shape[i] > 1.5*shape[i] and not bypass_vol_error:
      raise ValueError("Volume is very large, 150% of input size. Bad Scale?")

  out_vols = []
  out_codes = []
  for x_idx, x_slice in enumerate(slices[0]):
    for y_idx, y_slice in enumerate(slices[1]):
      for z_idx, z_slice in enumerate(slices[2]):
        out_vols.append( resize_zero_padding(vol[ (x_slice,y_slice,z_slice) ], shape))
        out_codes.append((x_idx,y_idx,z_idx))

  return out_vols, out_codes

#
# 191029
#
# this function takes a set of subvolumes produced with get_subvolumes (and
# possibly processed by other functions) and returns them to the original_vol
# shape. Note that the order MUST be preserved between get_subvolumes and
# this function.
def sub_vols_to_original_shape(sub_vols, sub_codes, orig_shape, recomb="tile"):

  shape = sub_vols[0].shape
  slices = defaultdict(lambda : [])
  sub_slices = defaultdict(lambda : [])
  for i in range(3):
    if orig_shape[i] < shape[i]:
      slices[i].append(slice( 0, orig_shape[i] ) )
      dx = (shape[i] - orig_shape[i]) // 2
      diff_mod = (shape[i] - orig_shape[i]) % 2

      sub_slices[i].append(slice( dx, shape[i] - diff_mod - dx ))
      continue

    slices[i].append( slice(0,np.min((orig_shape[i],shape[i]))) )
    sub_slices[i].append( slice(0,shape[i]) )
    if orig_shape[i] > shape[i]:
      slices[i].append( slice(shape[i], orig_shape[i]) )
      sub_slices[i].append( slice(orig_shape[i] - shape[i]+1, shape[i]) )
    if orig_shape[i] > 1.5*shape[i] and not bypass_vol_error:
      raise ValueError("Volume is very large, 150% of input size. Bad Scale?")

  output_vol = np.nan * np.ones(orig_shape)
  for code, vol in zip(sub_codes,sub_vols):
    output_vol[ slices[0][code[0]], slices[1][code[1]], slices[2][code[2]] ] \
      = vol[ sub_slices[0][code[0]], sub_slices[1][code[1]], sub_slices[2][code[2]] ]
  return output_vol





def preprocess(
    files,
    resize=None, #tuple of desired shape
    top_clip_percent=None, #float in [0,1] of desired top value, preceeds rescale
    bot_clip_percent=None, #float in [0,1] of desired bot value, preceeds rescale
    rescale_percentile=None, #float in [0,1] of desired percentile
  ):

  if isinstance(files, str):
    vol = read_vol(files)
  else:
    vol = np.concatenate([read_vol(f) for f in files], axis=-1)

  percentile_list = []

  if top_clip_percent is not None:
    percentile_list.append(top_clip_percent)
  else:
    percentile_list.append(100)

  if bot_clip_percent is not None:
    percentile_list.append(bot_clip_percent)
  else:
    percentile_list.append(0)

  #rescale and clip
  if rescale_percentile is not None:
    #TODO: type check here
    percentile_list.append(rescale_percentile)
  else:
    #OLD rescale
    percentile_list.append(100)

  top,bot,rescale_val = np.percentile(vol.flat,percentile_list)
  vol = np.clip(vol, a_min=bot, a_max=top)
  vol = vol / rescale_val

  if resize is not None:
    vol = resize_zero_padding_nd(vol, resize,)

  return vol


