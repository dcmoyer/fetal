

import numpy as np
from numpy import fft as npfft
import nibabel as nib
import constants
import new_labels
import os
N_freqs = 50

def freq_downsample( img_block, num_freqs ):
  img_block_mean = img_block.mean()
  img_block = img_block - img_block_mean

  ##
  ## first fft
  ##
  fft_block = npfft.fftn(img_block)
  fft_block = npfft.fftshift(fft_block)

  ##
  ##
  ##
  
  fft_mask = np.zeros(fft_block.shape)
  dim_x = fft_block.shape[0]
  dim_y = fft_block.shape[1]
  dim_z = fft_block.shape[2]
  center_x = dim_x // 2
  center_y = dim_y // 2
  center_z = dim_z // 2
  num_x = num_freqs // 2
  num_y = num_freqs // 2
  num_z = num_freqs // 2
  fft_mask[ \
    center_x - num_x:center_x + num_x, \
    center_y - num_y:center_y + num_y, \
    center_z - num_z:center_z + num_z \
  ] = 1

  ##
  ##
  ##

  fft_block = fft_block * fft_mask

  fft_recon = npfft.ifftshift(fft_block)
  fft_recon = npfft.ifftn(fft_recon)
  fft_recon += img_block_mean

  return fft_recon

path = "data/downsampled_%i/" % N_freqs
os.makedirs(path, exist_ok=True)

#input_file_format = "/data/vision/polina/projects/fetal_segmentation/data/rawdata/{s}/{s}_{n}.nii.gz"
#for subj in constants.LABELED_SAMPLES:
for subj in new_labels.NEW_LABELS.keys():
  target_frames = new_labels.NEW_FRAMES[subj]

  os.makedirs(path + subj, exist_ok=True)

  for frame in target_frames:
    
    filename = "/data/vision/polina/projects/fetal_segmentation/data/rawdata/%s/%s_%0.4i.nii.gz" % (subj,subj,frame)
    scan = nib.load(filename)
    scan_affine = scan.affine
    scan_block = scan.get_fdata()

    print(scan_block.shape)

    img_block = freq_downsample( scan_block, 50 )
    output_scan = nib.Nifti1Image(img_block.astype(np.float32), scan_affine)

    out_path = path + subj + "/%s_%0.4i.nii.gz" % (subj, frame)
    nib.save( output_scan, out_path )

  print(subj)





