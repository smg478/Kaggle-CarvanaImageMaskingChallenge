# Thanks to Peter Giannakopoulos https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523

import pandas as pd
import numpy as np
import cv2
from time import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import optimizers
from u_net import get_unet1_1024x1024


df_train = pd.read_csv('input/train_masks_weighted.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])


input_h = 1024 #640 #1280  #640 832x1248
input_w = 1024 #960 #1920  #960

epochs = 50
batch_size = 4

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=2017) 

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

####################################################################################################
# Augmentation methods

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomHueSaturationValue(image, mask, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        mask = mask

    return image, mask


def gamma_correction(img, limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = img/255.0
        img = cv2.pow(img, alpha)
    return np.uint8(img*255)


def equalize_hist(img):
    for c in range(0, 2):
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #img[:,:,c] = clahe.apply(img[:,:,c])
        img[:,:,c] = cv2.equalizeHist(img[:,:,c])
    return img


def random_contrast(img, limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        img = img / 255.0
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return np.uint8(img*255)


def random_brightness(img, limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = img / 255.0
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return np.uint8(img*255)


def random_saturation(img, limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        img = img / 255.0
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return np.uint8(img*255)


def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        img = img / 255.0
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return np.uint8(img*255)
  
#####################################################################################################
# data preparation

def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread('input/train_hq/{}.jpg'.format(id))
                img = cv2.resize(img, (input_w, input_h))

                # Augmentation Testing ==================================================
                #img = gamma_correction(img, u=0.5)
                #img = equalize_hist(img)
                #img = cv2.blur(img, (3, 3))
                #img = cv2.bilateralFilter(img, 3, 40, 40)
                #img = img[160:1120, 0:960]  # NOTE: its img[y: y + h, x: x + w]
                #img = random_saturation(img, u=0.25)
                #img = random_brightness(img, u=0.25)
                #img = random_contrast(img, u=0.25)
                #img = random_gray(img, u=0.25)
                #===============================================================

                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_w, input_h))
                
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.0, 0.0),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                img, mask = randomHueSaturationValue(img, mask,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))

                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread('input/train_hq/{}.jpg'.format(id))
                img = cv2.resize(img, (input_w, input_h))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_w, input_h))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=2,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               patience=3,
                               cooldown=2,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/unet1_{epoch:02d}-{dice_coeff:.5f}-{val_dice_coeff:.5f}.hdf5',
                             save_best_only=False,
                             save_weights_only=True,
                             period = 1),
             TensorBoard(log_dir="logs/")]


model = get_unet1_1024x1024()


print('model created.')
model.load_weights(filepath='weights/best_weights/unet1_05-0.99657-0.99641.hdf5', by_name=True)
print('Weights loaded.')


model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
