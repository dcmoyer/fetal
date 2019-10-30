

import constants
import numpy as np
from image3d import ImageTransformer, Iterator
#from process import preprocess
import simple_preproc
import nibabel as nib

def _format(file_formats, s, n):
    if isinstance(file_formats, str):
        return file_formats.format(s=s, n=str(n).zfill(4), p=str(max(0, n-1)).zfill(4))
    else:
        files = []
        for f in file_formats:
            files.append(f.format(s=s, n=str(n).zfill(4), p=str(max(0, n-1)).zfill(4)))
        return files

#
# frames is a dictionary with
#   sample_handles as keys 
#   lists of frame_indices as values
class DataGenerator(Iterator):

    #
    # init defines a function handle variable
    #   self.preproc_func
    # currently this is either set to preprocess( [x], resize=self.resize) or
    # an identity function if the preprocessing is called before-hand.
    def __init__(self,
                 list_of_samples,
                 frames=None,
                 input_file_format=None,
                 label_file_format=None,
                 label_types=None,
                 load_files=True,
                 random_gen=False,
                 augment=False,
                 resize=None,
                 top_clip=None,
                 rescale_percent=None,
                 batch_size=1,
                 seed=None,
                 input_file_list=None,
                 label_file_list=None,
                 ):

        self.frames = frames
        self.samples = list_of_samples
        self.load_files = load_files
        self.random_gen = random_gen
        self.augment = augment

        self.resize = resize
        self.top_clip = top_clip
        self.rescale_percent = rescale_percent

        self.input_file_format = input_file_format
        self.label_file_format = label_file_format
        self.input_files = []
        self.label_files = None if self.label_file_format is None else []
        self.label_types = label_types

        if self.input_file_format is not None:
            if self.frames is None:
                raise ValueError("[simple_data_utils] for input_file_format frames cannot be NoneType")
            for s in self.frames:
                for n in self.frames[s]:
                    self.input_files.append(_format(self.input_file_format, s, n))
                    if self.label_file_format:
                        self.label_files.append(_format(self.label_file_format, s, n))
        elif self.input_file_list is not None:
            self.input_files = self.input_file_list
            if self.label_file_list is not None:
                if len(self.label_file_list) != self.input_file_list:
                    raise ValueError("[simple_data_utils] label_file_list must match input_file_list")
                self.label_files = self.label_file_list
        else:
            raise ValueError("[simple_data_utils] either {input,label}_file_format or {input,label}_file_list must be present.")
            exit(1)

        self.inputs = self.input_files
        self.labels = self.label_files

        self.preproc_func = lambda x: simple_preproc.preprocess(
           x, resize=self.resize, top_clip_percent=self.top_clip, rescale_percentile=self.rescale_percent)
        self.preproc_func_labels = lambda x: simple_preproc.preprocess(
           x, resize=self.resize)

        if self.load_files:
            #this loads everything into memory
            print("[simple_data_utils] Preloading.")

            self.inputs = []
            for filename in self.input_files:
              self.inputs.append(self.preproc_func(filename))
            if self.label_files is not None:
              self.labels = []
              for filename in self.label_files:
                self.labels.append(self.preproc_func_labels(filename))

            #self.inputs = [self.preproc_func(file) for file in self.input_files]
            #if self.label_files is not None:
            #    self.labels = [self.preproc_func_labels(file) for file in self.label_files]
            self.preproc_func = lambda x: x
            self.preproc_func_labels = lambda x: x

        if self.augment:
            self.image_transformer = ImageTransformer(rotation_range=90.,
                                                      shift_range=0.1,
                                                      shear_range=0.1,
                                                      zoom_range=0.1,
                                                      crop_size=None,
                                                      fill_mode='nearest',
                                                      cval=0,
                                                      flip=True)

        super().__init__(len(self.inputs), batch_size, self.augment, seed)

    def _get_batch(self, index_array):

        batch = []
        if self.label_types is None:
            for _, i in enumerate(index_array):
                sample_idx = i
                if self.random_gen:
                    sample_idx = np.random.choice(len(self.inputs))

                #if load_files is true, this is an identity function, otherwise it's the 
                #preprocess function with preset args.
                x = self.preproc_func(self.inputs[i])

                if self.augment:
                    x = self.image_transformer.random_transform(x, seed=self.seed)
                batch.append(x)

            return np.asarray(batch)

        labels = []
        for _, i in enumerate(index_array):
            sample_idx = i
            if self.random_gen:
                sample_idx = np.random.choice(len(self.inputs))

            #if load_files is true, this is an identity function, otherwise it's the 
            #preprocess function with preset args
            x = self.preproc_func(self.inputs[sample_idx])
            y = self.preproc_func_labels(self.labels[sample_idx])

            if self.augment:
                x, y = self.image_transformer.random_transform(x, y, seed=self.seed)
            batch.append(x)
            labels.append(y)

        all_labels = []
        for label_type in self.label_types:
            if label_type == 'label':
                if self.labels is None:
                    raise ValueError('Labels not provided.')
                all_labels.append(labels)
            elif label_type == 'input':
                all_labels.append(batch)
            else:
                raise ValueError(f'Label type {label_type} is not supported.')

        if len(all_labels) == 1:
            all_labels = all_labels[0]

        return (np.asarray(batch), np.asarray(all_labels))



