import sys
from keras.models import Model
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D, Dense, Add, concatenate, Conv2D
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
#from metric import dice_coef, dice_coef_loss
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import binary_crossentropy
import keras.backend as K

IMG_ROWS, IMG_COLS = 640, 960

# metrics

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


# inception blocks

def _shortcut(_input, residual):
    stride_width = _input._keras_shape[1] / residual._keras_shape[1]
    stride_height = _input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == _input._keras_shape[3]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters = residual._keras_shape[3], kernel_size =  (1,1),
                                 strides=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(_input)

    return Add()([shortcut, residual])


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


def rblock(inputs, num, depth, scale=0.1):
    residual = Conv2D(depth, kernel_size = (num, num), border_mode='same')(inputs)
    residual = BatchNormalization()(residual)
    residual = Lambda(lambda x: x * scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)


def NConv2D(nb_filter, nb_row, nb_col, border_mode='same', strides=(1, 1)):
    def f(_input):
        conv = Conv2D(nb_filter=nb_filter, kernel_size =(nb_row, nb_col), strides=strides,
                             border_mode=border_mode)(_input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)

    return f


def BNA(_input):
    inputs_norm = BatchNormalization()(_input)
    return ELU()(inputs_norm)


def reduction_a(inputs, k=64, l=64, m=96, n=96):
    "35x35 -> 17x17"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(inputs_norm)

    conv2 = Conv2D(n, 3, 3, strides=(2, 2), border_mode='same')(inputs_norm)

    conv3_1 = NConv2D(k, 1, 1, strides=(1, 1), border_mode='same')(inputs_norm)
    conv3_2 = NConv2D(l, 3, 3, strides=(1, 1), border_mode='same')(conv3_1)
    conv3_2 = Conv2D(m, 3, 3, strides=(2, 2), border_mode='same')(conv3_2)

    res = concatenate([pool1, conv2, conv3_2], axis=3)
    return res


def reduction_b(inputs):
    "17x17 -> 8x8"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(inputs_norm)
    #
    conv2_1 = NConv2D(64, 1, 1, strides=(1, 1), border_mode='same')(inputs_norm)
    conv2_2 = Conv2D(96, 3, 3, strides=(2, 2), border_mode='same')(conv2_1)
    #
    conv3_1 = NConv2D(64, 1, 1, strides=(1, 1), border_mode='same')(inputs_norm)
    conv3_2 = Conv2D(72, 3, 3, strides=(2, 2), border_mode='same')(conv3_1)
    #
    conv4_1 = NConv2D(64, 1, 1, strides=(1, 1), border_mode='same')(inputs_norm)
    conv4_2 = NConv2D(72, 3, 3, strides=(1, 1), border_mode='same')(conv4_1)
    conv4_3 = Conv2D(80, 3, 3, strides=(2, 2), border_mode='same')(conv4_2)
    #
    res = concatenate([pool1, conv2_2, conv3_2, conv4_3], axis=3)
    return res


def get_unet_inception_2head(optimizer):
    splitted = True
    act = 'elu'

    inputs = Input((1, IMG_ROWS, IMG_COLS), name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)
    # conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)

    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = NConvolution2D(256, 3, 3, border_mode='same', subsample=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    # conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)

    #
    pre = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)
    #

    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)

    after_conv3 = rblock(conv3, 1, 128)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), after_conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, splitted=splitted, activation=act)
    conv7 = Dropout(0.5)(conv7)

    after_conv2 = rblock(conv2, 1, 64)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), after_conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up8, 64, batch_mode=2, splitted=splitted, activation=act)
    conv8 = Dropout(0.5)(conv8)

    after_conv1 = rblock(conv1, 1, 32)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), after_conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, splitted=splitted, activation=act)
    # conv9 = inception_block(conv9, 32, batch_mode=2, splitted=splitted, activation=act)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid', name='main_output')(conv9)
    # print conv10._keras_shape

    model = Model(input=inputs, output=[conv10, aux_out])
    model.compile(optimizer=optimizer,
                  loss={'main_output': dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice_coef, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


def get_unet_inception():
    splitted = True
    act = 'elu'

    # 640
    inputs = Input((IMG_ROWS, IMG_COLS,3), name='main_input')
    conv1 = inception_block(inputs, 32, splitted=splitted, activation=act)
    # conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = NConv2D(32, 3, 3, border_mode='same', strides=(1, 1))(pool1)
    #pool1 = Dropout(0.5)(pool1)

    conv2 = inception_block(pool1, 64, splitted=splitted, activation=act)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = NConv2D(64, 3, 3, border_mode='same', strides=(1, 1))(pool2)
    #pool2 = Dropout(0.5)(pool2)

    conv3 = inception_block(pool2, 128, splitted=splitted, activation=act)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = NConv2D(128, 3, 3, border_mode='same', strides=(1, 1))(pool3)
    #pool3 = Dropout(0.5)(pool3)

    conv4 = inception_block(pool3, 256, splitted=splitted, activation=act)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = NConv2D(256, 3, 3, border_mode='same', strides=(1, 1))(pool4)
    #pool4 = Dropout(0.5)(pool4)

    conv5 = inception_block(pool4, 512, splitted=splitted, activation=act)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = NConv2D(512, 3, 3, border_mode='same', strides=(1, 1))(pool5)
    # conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
    #conv5 = Dropout(0.5)(conv5)

    #
    #pre = Conv2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv5)
    #pre = Flatten()(pre)
    #aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)
    #

    after_conv5 = rblock(conv5, 1, 512)
    up5 = concatenate([UpSampling2D(size=(2, 2))(pool5), after_conv5], axis=3)
    conv5 = inception_block(up5, 512, splitted=splitted, activation=act)
    #conv6 = Dropout(0.5)(conv6)

    after_conv4 = rblock(conv4, 1, 256)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=3)
    conv6 = inception_block(up6, 256, splitted=splitted, activation=act)
    #conv6 = Dropout(0.5)(conv6)

    after_conv3 = rblock(conv3, 1, 128)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=3)
    conv7 = inception_block(up7, 128, splitted=splitted, activation=act)
    #conv7 = Dropout(0.5)(conv7)

    after_conv2 = rblock(conv2, 1, 64)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=3)
    conv8 = inception_block(up8, 64, splitted=splitted, activation=act)
    #conv8 = Dropout(0.5)(conv8)

    after_conv1 = rblock(conv1, 1, 32)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=3)
    conv9 = inception_block(up9, 32, splitted=splitted, activation=act)
    # conv9 = inception_block(conv9, 32, batch_mode=2, splitted=splitted, activation=act)
    #conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), init='he_normal', activation='sigmoid', name='main_output')(conv9)
    # print conv10._keras_shape

    #model = Model(input=inputs, output=[conv10, aux_out])
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=0.001),
                  loss={'main_output': bce_dice_loss},
                  #loss={'main_output': bce_dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice_coeff})
                  #loss_weights={'main_output': 1., 'aux_output': 0.5},
                  #loss_weights={'main_output': 1., 'aux_output': 0.5})

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def global_conv_block(inputs, k=31, channel=16):
    feature_1 = Conv2D(channel, (k, 1), activation='relu', padding='same', dilation_rate=(1, 1))(inputs)
    feature_1 = BatchNormalization()(feature_1)
    feature_1 = Conv2D(channel, (1, k), activation='relu', padding='same', dilation_rate=(1, 1))(feature_1)
    feature_1 = BatchNormalization()(feature_1)

    feature_2 = Conv2D(channel, (1, k), activation='relu', padding='same', dilation_rate=(1, 1))(inputs)
    feature_2 = BatchNormalization()(feature_2)
    feature_2 = Conv2D(channel, (k, 1), activation='relu', padding='same', dilation_rate=(1, 1))(feature_2)
    feature_2 = BatchNormalization()(feature_2)

    feature = Add()([feature_1, feature_2])
    return feature


def boundary_refine(inputs, channel=16):
    br = Conv2D(channel, (3, 3), padding='same', use_bias=True)(inputs)
    br = BatchNormalization()(br)
    br = Activation('relu')(br)
    br = Conv2D(channel, (3, 3), padding='same', use_bias=True)(br)
    return br
def get_unet_640_960_12(input_shape=(640, 960, 3),
                       num_classes=1):
    inputs = Input(shape=input_shape)

    input_conv = Conv2D(24, (3, 3), padding='same')(inputs)
    input_conv = BatchNormalization()(input_conv)
    input_conv= Activation('relu')(input_conv)

    # 640x960 -
    down0a = global_conv_block(input_conv, k=31, channel=16)
    down0a = Conv2D(24, (1, 1), padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    res_stream = Add()([down0a, input_conv]) #640

    # 320x480
    down0 = global_conv_block(down0a_pool, k=15, channel=32)
    down0 = Conv2D(24, (1, 1), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    res0 = UpSampling2D((2,2))(down0)
    res_stream = Add()([res0, res_stream])  #640

    # 160x240
    res_stream_pool =  MaxPooling2D((4, 4), strides=(4, 4))(res_stream)

    down1 = concatenate([down0_pool, res_stream_pool], axis=3)
    down1 = global_conv_block(down1, k=7, channel=64)
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
    up1 = global_conv_block(up1, k=7, channel=64)
    up1 = boundary_refine(up1, channel=64)
    up1 = Conv2D(24, (1, 1), padding='same')(up1)

    res_up1 = UpSampling2D((4, 4))(up1)
    res_stream = Add()([res_up1, res_stream])  # 640


    # 320x480
    res_stream_pool = MaxPooling2D((2, 2), strides=(2, 2))(res_stream)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([res_stream_pool, up0, down0], axis=3)
    up0 = global_conv_block(up0, k=15, channel=32)
    up0 = boundary_refine(up0, channel=32)
    up0 = Conv2D(24, (1, 1), padding='same')(up0)

    res_up0 = UpSampling2D((2, 2))(up0)
    res_stream = Add()([res_up0, res_stream])  # 640

    # 640x960

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([res_stream, down0a, up0a], axis=3)
    up0a = global_conv_block(up0a, k=31, channel=16)
    up0a = boundary_refine(up0a, channel=16)
    up0a = Conv2D(24, (1, 1), padding='same')(up0a)

    res_stream = Add()([up0a, res_stream])  # 640

    feature = global_conv_block(res_stream, k=31, channel=16)
    feature = boundary_refine(feature, channel=16)

    classify = Conv2D(1, (1, 1), activation='sigmoid', use_bias=True, name='classify_main')(feature)


    model = Model(inputs=inputs, outputs=classify)
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
