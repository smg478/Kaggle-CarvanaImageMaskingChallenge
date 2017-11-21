
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, Average, Add, SeparableConv2D, Conv2DTranspose
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import keras.backend as K
from keras import regularizers
#from keras.layers import BatchNormalization, Convolution2D, Input, merge
from keras.layers.core import Activation, Layer, SpatialDropout2D
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import PReLU, ELU
#from keras_contrib.layers import CRF
#from keras.preprocessing.sequence import pad_sequences

#from crfrnn_layer import CrfRnnLayer

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss


def inception_block(inputs, depth, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    c1_1 = Conv2D(filters = int(depth / 4), kernel_size = (1, 1), init='he_normal', border_mode='same')(inputs)

    c2_1 = Conv2D(filters = int(depth / 8 * 3), kernel_size = (1, 1), init='he_normal', border_mode='same')(inputs)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Conv2D(filters = int(depth / 2), kernel_size = (1, 3), init='he_normal', border_mode='same')(c2_1)
        c2_2 = BatchNormalization()(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(filters = int(depth / 2), kernel_size = (3, 1), init='he_normal', border_mode='same')(c2_2)
    else:
        c2_3 = Conv2D(filters=int(depth / 2), kernel_size=(3, 3), init='he_normal', border_mode='same')(c2_1)

    c3_1 = Conv2D(filters=int(depth / 16), kernel_size=(1, 1), init='he_normal', border_mode='same')(inputs)
    # missed batch norm
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Conv2D(filters=int(depth / 8), kernel_size=(1, 5), init='he_normal', border_mode='same')(c3_1)
        c3_2 = BatchNormalization()(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(filters=int(depth / 8), kernel_size=(5, 1), init='he_normal', border_mode='same')(c3_2)
    else:
        c3_3 = Conv2D(filters=int(depth / 8), kernel_size=(5, 5), init='he_normal', border_mode='same')(c3_1)

    p4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(inputs)
    c4_2 = Conv2D(filters=int(depth / 8), kernel_size=(1, 1), init='he_normal', border_mode='same')(p4_1)

    res = concatenate([c1_1, c2_3, c3_3, c4_2], axis=3)
    res = BatchNormalization()(res)
    res = actv()(res)
    return res


def global_conv_block(inputs, k=31, channel=16):

    feature_1 = Conv2D(channel, (k, 1), activation='relu', padding='same', dilation_rate=(1, 1))(inputs)
    feature_1 = BatchNormalization()(feature_1)
    feature_1 = Conv2D(channel, (1, k), activation='relu', padding='same', dilation_rate=(1, 1))(feature_1)
    feature_1 = BatchNormalization()(feature_1)

    feature_2 = Conv2D(channel, (1, k), activation='relu', padding='same', dilation_rate=(1, 1))(inputs)
    feature_2 = BatchNormalization()(feature_2)
    feature_2 = Conv2D(channel, (k, 1), activation='relu', padding='same', dilation_rate=(1, 1))(feature_2)
    feature_2 = BatchNormalization()(feature_2)

    feature = Add()([feature_1,feature_2])
    return feature


def boundary_refine(inputs, channel=16):
    br = Conv2D(channel, (3, 3), padding='same', use_bias=True)(inputs)
    br = BatchNormalization()(br)
    br = Activation('relu')(br)
    br = Conv2D(channel, (3, 3), padding='same', use_bias=True)(br)
    return br

########################################################################################

def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=[dice_loss])
    #model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model

def get_unet_256(input_shape=(256, 256, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model

def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])

    return model

########################################################################################

def get_funet_640_960(input_shape=(640, 960, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)


    # ===== Fractal block ==========

    # blk 1 640
    x11 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x11 = BatchNormalization()(x11)
    x21 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x11)
    x21 = BatchNormalization()(x21)
    avg1 = Average()([x11, x21])
    x31 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(avg1)
    x31 = BatchNormalization()(x31)
    x41 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x31)
    x41 = BatchNormalization()(x41)
    blk1 = Average()([x11, x31, x41])

    blk1 = Activation('relu')(blk1)
    blk1_pool = MaxPooling2D((2, 2), strides=(2, 2))(blk1)

    # blk 2 320
    x12 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(blk1_pool)
    x12 = BatchNormalization()(x12)
    x22 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x12)
    x22 = BatchNormalization()(x22)
    avg2 = Average()([x12, x22])
    x32 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(avg2)
    x32 = BatchNormalization()(x32)
    x42 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x32)
    x42 = BatchNormalization()(x42)
    blk2 = Average()([x12, x32, x42])

    blk2 = Activation('relu')(blk2)
    blk2_pool = MaxPooling2D((2, 2), strides=(2, 2))(blk2)

    # blk 3 160
    x13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(blk2_pool)
    x13 = BatchNormalization()(x13)
    x23 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x13)
    x23 = BatchNormalization()(x23)
    avg3 = Average()([x13, x23])
    x33 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(avg3)
    x33 = BatchNormalization()(x33)
    x43 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x33)
    x43 = BatchNormalization()(x43)
    blk3 = Average()([x13, x33, x43])

    blk3 = Activation('relu')(blk3)
    blk3_pool = MaxPooling2D((2, 2), strides=(2, 2))(blk3)

    # Center 80
    center = Conv2D(128, (3, 3), padding='same')(blk3_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(128, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)

    # blk up 3 160
    blk3u = UpSampling2D((2, 2))(center)
    blk3u = concatenate([blk3, blk3u], axis=3)

    y13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(blk3u)
    y13 = BatchNormalization()(y13)
    y23 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(y13)
    y23 = BatchNormalization()(y23)
    avg3u = Average()([y13, y23])
    y33 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(avg3u)
    y33 = BatchNormalization()(y33)
    y43 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(y33)
    y43 = BatchNormalization()(y43)
    blk3u = Average()([y13, y33, y43])

    blk3u = Activation('relu')(blk3u)

    # blk up 2 320
    blk2u = UpSampling2D((2, 2))(blk3u)
    blk2u = concatenate([blk2, blk2u], axis=3)

    y12 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(blk2u)
    y12 = BatchNormalization()(y12)
    y22 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(y12)
    y22 = BatchNormalization()(y22)
    avg2u = Average()([y12, y22])
    y32 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(avg2u)
    y32 = BatchNormalization()(y32)
    y42 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(y32)
    y42 = BatchNormalization()(y42)
    blk2u = Average()([y12, y32, y42])

    blk2u = Activation('relu')(blk2u)

    # blk up 1 640
    blk1u = UpSampling2D((2, 2))(blk2u)
    blk1u = concatenate([blk1, blk1u], axis=3)

    y11 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(blk1u)
    y11 = BatchNormalization()(y11)
    y21 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(y11)
    y21 = BatchNormalization()(y21)
    avg1u = Average()([y11, y21])
    y31 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(avg1u)
    y31 = BatchNormalization()(y31)
    y41 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(y31)
    y41 = BatchNormalization()(y41)
    blk1u = Average()([y11, y31, y41])

    blk1u = Activation('relu')(blk1u)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(blk1u)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=[dice_loss])

    return model

def get_dialatedNet_640_960(input_shape=(640, 960, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(24, (3, 3), activation='relu', padding='same', dilation_rate=(1, 1), name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='same', dilation_rate=(1, 1),name='block1_conv2')(x)
    x = BatchNormalization()(x)
    blk1 = x



    # Block 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2),name='block2_conv2')(x)
    x = BatchNormalization()(x)
    blk2 = x

    # Block 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=(4, 4),name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=(4, 4),name='block3_conv2')(x)
    x = BatchNormalization()(x)
    #x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=(4, 4),name='block3_conv3')(x)
    #x = BatchNormalization()(x)
    blk3 = x


    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=(8, 8),name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=(8, 8),name='block4_conv2')(x)
    x = BatchNormalization()(x)
    #x = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=(8, 8),name='block4_conv3')(x)
    #x = BatchNormalization()(x)
    blk4 = x

    # Block 5
    x = Conv2D(8, (3, 3), activation='relu', padding='same', dilation_rate=(16, 16),name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', dilation_rate=(16, 16),name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', dilation_rate=(16, 16),name='block5_conv3')(x)
    x = BatchNormalization()(x)
    blk5 = x

    x = concatenate([blk1, blk2, blk3, blk4, blk5])

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', dilation_rate=(1, 1), name='classification')(x)

    model = Model(inputs=inputs, outputs=classify)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    model.compile(optimizer=Adam(lr=0.01), loss=bce_dice_loss, metrics=[dice_coeff])

    return model


#########################################################################################

def get_unet_640_960_8(input_shape=(640, 960, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    #                                  dilation_rate=(1, 1), activation=None, use_bias=True,
    #                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                                  kernel_constraint=None, bias_constraint=None)

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.0005),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model

def get_unet_1024_8(input_shape=(1024, 1024, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    #                                  dilation_rate=(1, 1), activation=None, use_bias=True,
    #                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                                  kernel_constraint=None, bias_constraint=None)

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.00015),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model

def get_unet_832x1216_8(input_shape=(832, 1216, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    #                                  dilation_rate=(1, 1), activation=None, use_bias=True,
    #                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                                  kernel_constraint=None, bias_constraint=None)

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.0002),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model

def get_unet_1280x1920_8(input_shape=(1280, 1920, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    #input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    #down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    #down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    #down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    #down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    #down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    #down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    #up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    #up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    #up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    #up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.0002),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model

def get_unet_1024x1536_8(input_shape=(1024, 1536, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    #                                  dilation_rate=(1, 1), activation=None, use_bias=True,
    #                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                                  kernel_constraint=None, bias_constraint=None)

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.00015),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model

def get_unet_1280x1280_8(input_shape=(1280, 1280, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    #                                  dilation_rate=(1, 1), activation=None, use_bias=True,
    #                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                                  kernel_constraint=None, bias_constraint=None)

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.0001),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model

def get_unet_1280x960_8(input_shape=(1280, 960, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    # Res block

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640


    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640



    # 160x240

    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4,4))(down1) #640
    res_stream = Add()([res1, res_stream])



    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1,tower2],axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1,tower2],axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    #up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    #                                  dilation_rate=(1, 1), activation=None, use_bias=True,
    #                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                                  kernel_constraint=None, bias_constraint=None)

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    #8 net experiment
    #features2 = Conv2D(8, (5, 5), padding='same', use_bias=True, name = 'featurs2')(res_stream)
    #classify2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify2')(features2)
    #features3 = Conv2D(8, (7, 7), padding='same', use_bias=True, name = 'featurs3')(res_stream)
    #classify3 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify3')(features3)

    #main_feature = Average()([classify1,classify2,classify3])
    #main_output = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='main_output')(main_feature)

    ##crf = CRF( units=128,sparse_target=True, activation='sigmoid',name = 'crf')(classify) # for word embedding. one dimentional

    #output = CrfRnnLayer(image_dims=(640, 960),
    #                     num_classes=1,
    #                     theta_alpha=160.,
    #                     theta_beta=3.,
    #                     theta_gamma=3.,
    #                     num_iterations=5,
    #                     name='crfrnn')([classify, inputs])

    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    model.compile(optimizer=Adam(lr=0.0001),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


##########################################################################################

def get_unet_640_960_11(input_shape=(640, 960, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)


    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv])  # 640

    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2, 2))(down0)
    res_stream = Add()([res0, res_stream])  # 640

    # 160x240

    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4, 4))(down1)  # 640
    res_stream = Add()([res1, res_stream])

    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1, tower2], axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1, tower2], axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    # up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640

    feature_1 = Conv2D(16, (31, 1), activation='relu', padding='same', dilation_rate=(1, 1))(res_stream)
    feature_1 = BatchNormalization()(feature_1)
    feature_1 = Conv2D(16, (1, 31), activation='relu', padding='same', dilation_rate=(1, 1))(feature_1)
    feature_1 = BatchNormalization()(feature_1)

    feature_2 = Conv2D(16, (1, 31), activation='relu', padding='same', dilation_rate=(1, 1))(res_stream)
    feature_2 = BatchNormalization()(feature_2)
    feature_2 = Conv2D(16, (31, 1), activation='relu', padding='same', dilation_rate=(1, 1))(feature_2)
    feature_2 = BatchNormalization()(feature_2)

    feature = Add()([feature_1, feature_2])

    main_feature = Conv2D(16, (3, 3), padding='same', use_bias=True, name = 'feature_main_cnv1')(feature)
    main_feature = BatchNormalization()(main_feature)
    main_feature = Activation('relu')(main_feature)
    main_feature = Conv2D(16, (3, 3), padding='same', use_bias=True, name='feature_main_cnv2')(main_feature)

    classify_main = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify_main')(main_feature)


    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify_main)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    model.compile(optimizer=Adam(lr=0.0005),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'classify_main': 1., 'classify_edge': 0.2,  'classify_res': 0.2})

    return model

def get_unet_832x1216_11(input_shape=(832, 1216, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)


    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower1 = down0a
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    tower2 = down0a
    down0a = concatenate([tower1, tower2], axis=3)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv])  # 640

    # 320x480
    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower1 = down0
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    tower2 = down0
    down0 = concatenate([tower1, tower2], axis=3)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2, 2))(down0)
    res_stream = Add()([res0, res_stream])  # 640

    # 160x240

    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower1 = down1
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    tower2 = down1
    down1 = concatenate([tower1, tower2], axis=3)
    down1 = Conv2D(24, (1, 1), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    res1 = UpSampling2D((4, 4))(down1)  # 640
    res_stream = Add()([res1, res_stream])

    # 80x120
    res_stream_pool =  MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    down2 = concatenate([down1_pool, res_stream_pool], axis=3)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower1 = down2
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    tower2 = down2
    down2 = concatenate([tower1, tower2], axis=3)
    down2 = Conv2D(24, (1, 1), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    res2 = UpSampling2D((8,8))(down2)
    res_stream = Add()([res2, res_stream])  #640



    # 40x60
    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)
    down3 = concatenate([down2_pool, res_stream_pool], axis=3)

    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower1 = down3
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    tower2 = down3
    down3 = concatenate([tower1, tower2], axis=3)
    down3 = Conv2D(24, (1, 1), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    res3 = UpSampling2D((16,16))(down3)
    res_stream = Add()([res3, res_stream])  #640

    # 20x30
    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    down4 = concatenate([down3_pool, res_stream_pool], axis=3)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower1 = down4
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    tower2 = down4
    down4 = concatenate([tower1, tower2], axis=3)
    down4 = Conv2D(24, (1, 1), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    res4 = UpSampling2D((32,32))(down4)
    res_stream = Add()([res4, res_stream])  #640


    # 10x15
    res_stream_pool = MaxPooling2D((64, 64), strides=(64, 64))(res_stream)

    center = concatenate([down4_pool, res_stream_pool], axis=3)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(24, (1, 1), padding='same')(center)

    resCen = UpSampling2D((64,64))(center)
    res_stream = Add()([resCen, res_stream])  #640

    # 20x30

    res_stream_pool = MaxPooling2D((32, 32), strides=(32, 32))(res_stream)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([res_stream_pool, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower1 = up4
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    tower2 = up4
    up4 = concatenate([tower1,tower2], axis=3)
    up4 = Conv2D(24, (1, 1), padding='same')(up4)

    res_up4 = UpSampling2D((32, 32))(up4)
    res_stream = Add()([res_up4, res_stream])  # 640


    # 40x60

    res_stream_pool = MaxPooling2D((16, 16), strides=(16, 16))(res_stream)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([res_stream_pool, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower1 = up3
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    tower2 = up3
    up3 = concatenate([tower1, tower2], axis=3)
    up3 = Conv2D(24, (1, 1), padding='same')(up3)

    res_up3 = UpSampling2D((16, 16))(up3)
    res_stream = Add()([res_up3, res_stream])  # 640


    # 80x120
    res_stream_pool = MaxPooling2D((8, 8), strides=(8, 8))(res_stream)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([res_stream_pool, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower1 = up2
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    tower2 = up2
    up2 = concatenate([tower1,tower2], axis=3)
    up2 = Conv2D(24, (1, 1), padding='same')(up2)

    res_up2 = UpSampling2D((8, 8))(up2)
    res_stream = Add()([res_up2, res_stream])  # 640

    # 160x240
    res_stream_pool = MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([res_stream_pool, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower1 = up1
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    tower2 = up1
    up1 = concatenate([tower1,tower2],axis=3)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower1 = up0
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    tower2 = up0
    up0 = concatenate([tower1, tower2], axis=3)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)

    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower1 = up0a
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    tower2 = up0a
    up0a = concatenate([tower1, tower2], axis=3)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)
    # up0a = SpatialDropout2D(0.5)(up0a)

    res_stream = Add()([up0a, res_stream])  # 640

    feature_1 = Conv2D(16, (31, 1), activation='relu', padding='same', dilation_rate=(1, 1))(res_stream)
    feature_1 = BatchNormalization()(feature_1)
    feature_1 = Conv2D(16, (1, 31), activation='relu', padding='same', dilation_rate=(1, 1))(feature_1)
    feature_1 = BatchNormalization()(feature_1)

    feature_2 = Conv2D(16, (1, 31), activation='relu', padding='same', dilation_rate=(1, 1))(res_stream)
    feature_2 = BatchNormalization()(feature_2)
    feature_2 = Conv2D(16, (31, 1), activation='relu', padding='same', dilation_rate=(1, 1))(feature_2)
    feature_2 = BatchNormalization()(feature_2)

    feature = Add()([feature_1, feature_2])

    main_feature = Conv2D(16, (3, 3), padding='same', use_bias=True, name = 'feature_main_cnv1')(feature)
    main_feature = BatchNormalization()(main_feature)
    main_feature = Activation('relu')(main_feature)
    main_feature = Conv2D(16, (3, 3), padding='same', use_bias=True, name='feature_main_cnv2')(main_feature)

    classify_main = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify_main')(main_feature)


    #model = Model(inputs=inputs, outputs=[output,classify])
    model = Model(inputs=inputs, outputs=classify_main)
    #model = Model(inputs=inputs, outputs=[classify,tower1,tower2,tower3])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    model.compile(optimizer=Adam(lr=0.0005),
                  loss=bce_dice_loss,
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics=[dice_coeff])
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'classify_main': 1., 'classify_edge': 0.2,  'classify_res': 0.2})

    return model

#########################################################################################3