# Thanks to Peter Giannakopoulos https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, Average, Add, SeparableConv2D, Conv2DTranspose
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import keras.backend as K
from keras import regularizers
from keras.layers.core import Activation, Layer, SpatialDropout2D
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import PReLU, ELU

#########################################################################################################
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

###########################################################################################################

def get_unet_basic_128(input_shape=(128, 128, 3),
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

    return model


def get_unet1_1024x1024(input_shape=(1024, 1024, 3),
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

    features = Conv2D(24, (3, 3), padding='same', use_bias=False)(res_stream)
    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True)(features)

    model = Model(inputs=inputs, outputs=classify)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    model.compile(optimizer=Adam(lr=0.00015),
                  loss=bce_dice_loss,
                  metrics=[dice_coeff])

    return model


###########################################################################################################

def get_unet2_832x1216(input_shape=(832, 1216, 3),
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


    model = Model(inputs=inputs, outputs=classify_main)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    model.compile(optimizer=Adam(lr=0.0005),
                  loss=bce_dice_loss,
                  metrics=[dice_coeff])

    return model

###########################################################################################
