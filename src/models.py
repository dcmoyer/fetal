

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
import os
#import process
import simple_preproc
import util

#TODO: move this elsewhere
import nibabel as nib

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def weighted_crossentropy(weights=None, boundary_weight=None, pool=5):
    w = (.5, .5) if weights is None else weights
    epsilon = K.epsilon()

    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy = K.stack([-(y_true * K.log(y_pred)), -((1 - y_true) * K.log(1 - y_pred))],
                                axis=-1)
        loss = w * cross_entropy

        if boundary_weight is not None:
            y_true_avg = K.pool3d(y_true, pool_size=(pool,)*3, padding='same', pool_mode='avg')
            boundaries = K.cast(y_true_avg > 0, 'float32') * K.cast(y_true_avg < 1, 'float32')
            loss += boundary_weight * K.stack([boundaries, boundaries], axis=-1) * cross_entropy

        return K.mean(K.sum(loss, axis=-1))
    return loss_fn


def save_prediction(pred, input_file, tile, path, scale=False, ):
    pass
    ###TODO: get rid of this fuckery
    ##fname = input_file.split('/')[-1]
    ##sample = fname.split('_')[0]
    ##path = os.path.join(path, sample)
    ##os.makedirs(path, exist_ok=True)
    ##shape = util.shape(input_file)
    ##header = util.header(input_file)

    ###vol = process.postprocess(pred, shape, resize=True, tile=tile)
    ##vol = simple_preproc.preprocess(pred)

    ##print(vol.shape)
    ##for i in range(3):
    ##  if vol.shape[i] > 

    ##util.save_vol(vol, os.path.join(path, fname), header, scale)
    ##print(fname, flush=True)


class BaseModel:
    def __init__(self, input_size, name=None, filename=None, weights=None,
        log_location='logs', save_location='models', output_location='data/predict'):

        self.log_location = log_location
        self.save_location = save_location
        self.output_location = output_location

        self.input_size = input_size
        self.name = name if name else self.__class__.__name__.lower()
        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)
        self._compile(weights)

    def _new_model(self):
        raise NotImplementedError()

    def _compile(self, weights):
        raise NotImplementedError()

    def train(self, generator, val_gen, epochs):
        path = f'{self.save_location}/{self.name}'
        
        os.makedirs(path, exist_ok=True)
        if val_gen:
            file_name = '{epoch:0>4d}_{val_dice_coef:.4f}.h5'
        else:
            file_name = '{epoch:0>4d}.h5'
        self.model.fit_generator(
          generator,
          epochs=epochs,
          validation_data=val_gen,
          verbose=1,
          callbacks=[ModelCheckpoint(os.path.join(path, file_name), save_weights_only=True, period=50),
          TensorBoard(log_dir=f'{self.log_location}/{self.name}')]
        )

    def predict(self, generator, subj_dirs=False):

        n = len(generator)
        for i in range(n):

            #TODO: why is this hard coded here.
            input_file = generator.input_files[i]

            subj_dir = input_file.split("/")[-2]

            path = f'{self.output_location}/{subj_dir}'
            os.makedirs(path, exist_ok=True)

            fname = input_file.split('/')[-1]
            header = util.header(input_file)

            #vol = nib.load(input_file)
            vol = generator[i]
            #vol = vol[0,:,:,:]
            sub_vols, sub_codes = simple_preproc.get_subvolumes(vol[0], self.input_size )
            preds = []
            for sub_vol in sub_vols:
                preds.append(self.model.predict( sub_vol[np.newaxis,:,:,:,:])[0])
            pred = simple_preproc.sub_vols_to_original_shape( 
              preds, sub_codes, vol.shape[1:]
            )
            util.save_vol(pred, os.path.join(path, fname), )

    def test(self, generator):
        metrics = self.model.evaluate_generator(generator)
        return dict(zip(self.model.metrics_names, [metrics] if isinstance(metrics, float) else metrics))

##    def predict(self, generator):
##        print("[models] TODO: replace hard-coded predict location")
##        path = f'{self.output_location}/{self.name}'
##        os.makedirs(path, exist_ok=True)
##
##        #tile = generator.tile_inputs
##        tile = False
##        n = len(generator)//8 if tile else len(generator)
##        for i in range(n):
##            input_file = generator.input_files[i]
##            if not isinstance(input_file, str):
##                input_file = input_file[0]
##            pred = self.model.predict(
##              np.concatenate([generator[8*i+j] for j in range(8)]) if tile else generator[i]
##            )
##            save_prediction(pred, input_file, tile, path)


class UNet(BaseModel):
    # 140 perceptive field
    def _new_model(self):

        #self.input_shape = (128,128,96,1)
        inputs = layers.Input(shape=self.input_size)
        #inputs = layers.Input(shape=(128,128,96,1))
        #inputs = layers.Input(shape=(None,None,None,1))

        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)
        print(conv4.shape)

        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
        print(conv5.shape)

        up6 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        print(up6.shape)
        conc6 = layers.concatenate([up6, conv4])
        print(conc6.shape)
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv3])
        print(conc7.shape)
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = layers.concatenate([up8, conv2])
        print(conc8.shape)
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = layers.concatenate([up9, conv1])
        print(conc9.shape)
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self, weights):
        self.model.compile(optimizer=Adam(lr=5e-5),
                           loss=weighted_crossentropy(weights=weights, boundary_weight=1.),
                           metrics=[dice_coef])


class UNetSmall(UNet):
    # 68 perceptive field
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)

        up5 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
        conc5 = layers.concatenate([up5, conv3])
        conv5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc5)
        conv5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        conc6 = layers.concatenate([up6, conv2])
        conv6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv1])
        conv7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', name='outputs')(conv7)

        self.model = Model(inputs=inputs, outputs=outputs)


class AESeg(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv4)

        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        conc6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = layers.concatenate([up9, conv1])
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', name='outputs')(conv9)

        ae_up6 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        ae_conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(ae_up6)
        ae_conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(ae_conv6)

        ae_up7 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(ae_conv6)
        ae_conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(ae_up7)
        ae_conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(ae_conv7)

        ae_up8 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(ae_conv7)
        ae_conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(ae_up8)
        ae_conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(ae_conv8)

        ae_up9 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(ae_conv8)
        ae_conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(ae_up9)
        ae_conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(ae_conv9)

        ae_outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', name='ae_outputs')(ae_conv9)

        self.model = Model(inputs=inputs, outputs=[outputs, ae_outputs])

    def _compile(self, weights):
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss={'outputs': weighted_crossentropy(weights=weights, boundary_weight=1.), 'ae_outputs': 'mse'},
                           loss_weights={'outputs': .5, 'ae_outputs': .5},
                           metrics={'outputs': dice_coef, 'ae_outputs': 'accuracy'})

    def predict(self, generator):        
        path = f'{self.output_location}/{self.name}'
        print("[models] TODO: replace hard-coded predict location")
        os.makedirs(path, exist_ok=True)

        for i in range(len(generator)):
            input_file = generator.input_files[i]
            tile = generator.tile_inputs
            pred, vol = self.model.predict(np.concatenate([generator[i+j] for j in range(8)]) if tile else generator[i])
            save_prediction(pred, input_file, tile, path)
            save_prediction(vol, input_file, tile, os.join(path, 'ae_reconstructions'), scale=True)
