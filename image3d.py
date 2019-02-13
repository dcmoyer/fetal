"""Fairly basic set of tools for real-time data augmentation on 3D image data.

Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial

from keras import backend as K
from keras.utils.data_utils import Sequence


def transform_matrix_offset_center(matrix, shape):
    o_x = float(shape[0]) / 2 + 0.5
    o_y = float(shape[1]) / 2 + 0.5
    o_z = float(shape[2]) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x],
                              [0, 1, 0, o_y],
                              [0, 0, 1, o_z],
                              [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x],
                             [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z],
                             [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=3,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 4D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:-1, :-1]
    final_offset = transform_matrix[:-1, -1]
    channel_images = [ndi.interpolation.affine_transform(
                      channel,
                      final_affine_matrix,
                      final_offset,
                      order=1,
                      mode=fill_mode,
                      cval=cval) for channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class ImageTransformer(object):
    """Transforms image data.

    # Arguments
        rotation_range: degrees (0 to 180).
        shift_range: fraction of each dimension.
        shear_range: shear intensity (fraction).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        flip: whether to randomly flip images along its axes.
    """

    def __init__(self,
                 rotation_range=0.,
                 shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 crop_size=None,
                 fill_mode='nearest',
                 cval=0.,
                 flip=False):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.crop_size = crop_size
        self.fill_mode = fill_mode
        self.cval = cval
        self.flip = flip

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

        if shear_range > 1:
            raise ValueError('`shear_range` should be a float. '
                             'Received arg: ', shear_range)

    def random_transform(self, x, y=None, seed=None):
        """Randomly augment a single image tensor and optionally its label.

        # Arguments
            x: 3D tensor, single image.
            y: 3D tensor, label of x. Must be the same shape as x.
            seed: random seed.

        # Returns
            A randomly transformed version of the input and label (same shape).
        """
        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        transform_matrix = None

        if self.rotation_range:
            rx, ry, rz = np.deg2rad(np.random.uniform(-self.rotation_range,
                                                      self.rotation_range,
                                                      3))
            Rx = np.array([[1, 0, 0, 0],
                           [0, np.cos(rx), -np.sin(rx), 0],
                           [0, np.sin(rx), np.cos(rx), 0],
                           [0, 0, 0, 1]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                           [0, 1, 0, 0],
                           [-np.sin(ry), 0, np.cos(ry), 0],
                           [0, 0, 0, 1]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                           [np.sin(rz), np.cos(rz), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
            rotation_matrix = np.dot(np.dot(Rx, Ry), Rz)
            transform_matrix = rotation_matrix

        if self.shift_range:
            tx, ty, tz = np.random.uniform(-self.shift_range, self.shift_range, 3)
            if self.shift_range < 1:
                tx *= x.shape[0]
                ty *= x.shape[1]
                tz *= x.shape[2]
            shift_matrix = np.array([[1, 0, 0, tx],
                                     [0, 1, 0, ty],
                                     [0, 0, 1, tz],
                                     [0, 0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if self.shear_range:
            sxy, sxz, syx, syz, szx, szy = np.random.uniform(-self.shear_range,
                                                             self.shear_range,
                                                             6)
            shear_matrix = np.array([[1, sxy, sxz, 0],
                                     [syx, 1, syz, 0],
                                     [szx, szy, 1, 0],
                                     [0, 0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if self.zoom_range[0] != 1 and self.zoom_range[1] != 1:
            zx, zy, zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)
            zoom_matrix = np.array([[zx, 0, 0, 0],
                                    [0, zy, 0, 0],
                                    [0, 0, zz, 0],
                                    [0, 0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            transform_matrix = transform_matrix_offset_center(transform_matrix, x.shape)
            x = apply_transform(x,
                                transform_matrix,
                                fill_mode=self.fill_mode,
                                cval=self.cval)
            if y is not None:
                y = apply_transform(y,
                                    transform_matrix,
                                    fill_mode=self.fill_mode,
                                    cval=self.cval)

        if self.crop_size:
            cx = np.random.randint(0, x.shape[0]-self.crop_size[0])
            cy = np.random.randint(0, x.shape[1]-self.crop_size[1])
            cz = np.random.randint(0, x.shape[2]-self.crop_size[2])
            x = x[cx:cx+self.crop_size[0], cy:cy+self.crop_size[1], cz:cz+self.crop_size[2], :]
            if y is not None:
                y = y[cx:cx+self.crop_size[0], cy:cy+self.crop_size[1], cz:cz+self.crop_size[2], :]

        if self.flip:
            for axis in range(3):
                if np.random.random() < 0.5:
                    x = flip_axis(x, axis)
                    if y is not None:
                        y = flip_axis(y, axis)

        return x if y is None else (x, y)


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        repeat = (self.n + self.batch_size - 1) // self.n
        if self.shuffle:
            self.index_array = np.ravel([np.random.permutation(self.n) for _ in range(repeat)])
        else:
            self.index_array = np.ravel([np.arange(self.n)] * repeat)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(f'Asked to retrieve element {idx}, but the Sequence has length {len(self)}')
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array, load_fn=None):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: array of sample indices to include in batch.
            load_fn: function for loading in data.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class VolumeIterator(Iterator):
    """Iterator yielding data for 3D volumes.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of label data.
        image_transformer: Instance of `ImageTransformer`
            to use for random transformations.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
    """

    def __init__(self, x, y, image_transformer,
                 batch_size=32, shuffle=True, seed=None):
        self.x = x
        if any(x[i].ndim != 4 for i in range(len(x))):
            raise ValueError('Each volume of the input data for '
                             '`VolumeIterator` should have rank 4.')
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.image_transformer = image_transformer

    def _get_batches_of_transformed_samples(self, index_array, load_fn=None):
        batch_x = []
        if self.y is None:
            for i, j in enumerate(index_array):
                if load_fn is None:
                    x = self.x[j]
                else:
                    x = load_fn(self.x[j])
                x = self.image_transformer.random_transform(x)
                batch_x.append(x)
            batch_x = np.asarray(batch_x, dtype=K.floatx())
            return batch_x

        batch_y = []
        for i, j in enumerate(index_array):
            if load_fn is None:
                x, y = self.x[j], self.y[j]
            else:
                x, y = load_fn(self.x[j]), load_fn(self.y[j])
            x, y = self.image_transformer.random_transform(x, y)
            batch_x.append(x)
            batch_y.append(y)
        return (np.asarray(batch_x, dtype=K.floatx()), np.asarray(batch_y, dtype=K.floatx()))

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
