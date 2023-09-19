
from typing import Callable, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input,
                                     MaxPooling2D, Permute, Reshape,
                                     SpatialDropout2D, UpSampling2D,
                                     concatenate)
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import L2
from tensorflow_addons.layers import InstanceNormalization

from src.utils.losses import dice_bce_loss


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal", l2=1e-4):
    x = Conv2D(
        filters=nfilters,
        kernel_size=(size, size),
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=L2(l2),
        bias_regularizer=L2(l2),
        )(tensor)
    x = InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters=nfilters,
        kernel_size=(size, size),
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=L2(l2),
        bias_regularizer=L2(l2),
        )(x)
    x = InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2), l2=1e-4):
    y = Conv2DTranspose(
        nfilters,
        kernel_size=(size, size),
        strides=strides,
        padding=padding,
        kernel_regularizer=L2(l2),
        bias_regularizer=L2(l2),
        )(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y

def Unet(img_height, img_width, nclasses=4, filters=8, do=0.2, l2=1e-4):
# down
    input_layer = Input(shape=(img_height, img_width, 1), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters, l2=l2)
    norm1 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(conv1)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(norm1)
    conv1_out = SpatialDropout2D(do)(conv1_out)
    conv2 = conv_block(conv1_out, nfilters=filters*2, l2=l2)
    norm2 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(conv2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(norm2)
    conv2_out = SpatialDropout2D(do)(conv2_out)
    conv3 = conv_block(conv2_out, nfilters=filters*4, l2=l2)
    norm3 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(conv3)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(norm3)
    conv3_out = SpatialDropout2D(do)(conv3_out)
    conv4 = conv_block(conv3_out, nfilters=filters*8, l2=l2)
    norm4 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(conv4)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(norm4)
    conv4_out = SpatialDropout2D(do)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16, l2=l2)
    norm5 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(conv5)
    conv5 = SpatialDropout2D(do)(norm5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8, l2=l2)
    norm6 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(deconv6)
    deconv6 = SpatialDropout2D(do)(norm6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4, l2=l2)
    norm7 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(deconv7)
    deconv7 = SpatialDropout2D(do)(norm7)
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2, l2=l2)
    norm8 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(deconv8)
    deconv8 = SpatialDropout2D(do)(norm8)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters, l2=l2)
    norm9 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )(deconv9)

# output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(norm9)
    output_layer = BatchNormalization()(output_layer)
    #output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')

    return model
