

import constants
import numpy as np
from util import read_vol
from scipy.ndimage.measurements import label


def crop(vol):
    dx = (vol.shape[0] - constants.SHAPE[0]) // 2
    dy = (vol.shape[1] - constants.SHAPE[1]) // 2
    dz = (vol.shape[2] - constants.SHAPE[2]) // 2

    resized = vol[dx:(dx+constants.SHAPE[0]),
                  dy:(dy+constants.SHAPE[1]),
                  dz:(dz+constants.SHAPE[2])]
    return resized


# dmoyer:
# this appears to pad with periodic boundaries.
# To avoid this set tile to be false
def split(vol):
    vols = []
    for i in slice(constants.SHAPE[0]), slice(-constants.SHAPE[0], vol.shape[0]):
        for j in slice(constants.SHAPE[1]), slice(-constants.SHAPE[1], vol.shape[1]):
            for k in slice(constants.SHAPE[2]), slice(-constants.SHAPE[2], vol.shape[2]):
                vols.append(vol[i, j, k])
    return np.asarray(vols)


def preprocess(files, resize=False, tile=False):
    if isinstance(files, str):
        vol = read_vol(files)
    else:
        vol = np.concatenate([read_vol(f) for f in files], axis=-1)
    vol = vol / np.max(vol, axis=(0,1,2))

    if (vol.shape[0] < constants.SHAPE[0] or
        vol.shape[1] < constants.SHAPE[1] or
        vol.shape[2] < constants.SHAPE[2] or
        vol.shape[0] > 2*constants.SHAPE[0] or
        vol.shape[1] > 2*constants.SHAPE[1] or
        vol.shape[2] > 2*constants.SHAPE[2]):
        raise ValueError(f'The input shape {vol.shape} is not supported.')

    if tile:
        vol = split(vol)
    elif resize:
        vol = crop(vol)
    return vol


def uncrop(vol, shape):
    dx = (shape[0] - vol.shape[0]) // 2
    dy = (shape[1] - vol.shape[1]) // 2
    dz = (shape[2] - vol.shape[2]) // 2

    resized = np.pad(vol, ((dx, shape[0] - vol.shape[0] - dx),
                           (dy, shape[1] - vol.shape[1] - dy),
                           (dz, shape[2] - vol.shape[2] - dz),
                           (0, 0)), 'constant')
    return resized

#
# 191021 
#
# This function takes the volume to the correct shape no matter what, zero padding or cropping
# to get there. 
#
# zeropadding comes first! then cropping.
def resize_zero_padding_nd(vol, shape, n_dims=3):
    #dx = (shape[0] - vol.shape[0]) // 2
    #dy = (shape[1] - vol.shape[1]) // 2
    #dz = (shape[2] - vol.shape[2]) // 2

    dxyz = []
    dxyz_mod = []
    for i in range(n_dims):
      if shape[i] is not None:
        dxyz.append((shape[i] - vol.shape[i]) // 2)
        dxyz_mod.append( (shape[i] - vol.shape[i]) % 2 )
      else:
        dxyz.append(None)
        dxyz_mod.append(None)

    crop_limits = []
    zpad_limits = []
    for i,(diff,diff_mod) in enumerate(zip(dxyz, dxyz_mod)):

      if diff is not None and diff > 0:
        # zero pad
        zpad_limits.append( (diff, shape[i] - vol.shape[i] + diff_mod) )
        crop_limits.append( (0,shape[i]) )
      elif diff is not None and diff < 0:
        # crop
        zpad_limits.append( (0,0) )
        crop_limits.append( (-diff,vol.shape[i] + diff + diff_mod) )
      else:
        # no change, either None or diff == 0
        zpad_limits.append( (0,0) )
        crop_limits.append( (0,shape[i]) )

    print(zpad_limits)
    print(crop_limits)

    resized = np.pad(vol, tuple(zpad_limits), 'constant')

    slices = tuple( slice(int(start),int(stop)) for start,stop in crop_limits )

    resized = resized[ slices ]

    return resized


def unsplit(vols, shape):
    vol = np.zeros(shape)
    mask = np.zeros(shape)

    dx = shape[0] - vols.shape[1]
    dy = shape[1] - vols.shape[2]
    dz = shape[2] - vols.shape[3]

    n = 0
    for i in (0, dx), (dx, 0):
        for j in (0, dy), (dy, 0):
            for k in (0, dz), (dz, 0):
                vol += np.pad(vols[n], (i, j, k, (0, 0)), 'constant')
                mask += np.pad(np.ones(vols[n].shape), (i, j, k, (0, 0)), 'constant')
                n += 1
    return np.rint(vol / mask).astype(int)


def postprocess(vol, shape, resize=False, tile=False):
    if vol.shape[-4:] != constants.SHAPE:
        raise ValueError(f'The volume shape {vol.shape} is not supported.')
    if (shape[-4] < constants.SHAPE[0] or
        shape[-3] < constants.SHAPE[1] or
        shape[-2] < constants.SHAPE[2]):
        raise ValueError(f'The target shape {shape} is not supported.')

    if tile:
        vol = unsplit(vol, shape)
    elif resize:
        vol = uncrop(vol, shape)
    return vol


def remove_artifacts(vol, n):
    assert n > 0
    cleaned = np.zeros(vol.shape)
    artifacts, _ = label(vol)
    indices = np.argsort(np.bincount(artifacts.flat))[-n-1:-1]
    for i in indices:
        cleaned += artifacts == i
    return cleaned
