import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from u_net import get_unet1_1024x1024


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

input_h = 640 #640 #1280  #640
input_w = 960 #960 #1920  #960

batch_size = 1
threshold = 0.5

model = get_unet1_1024x1024()
model.load_weights(filepath='weights/unet-1024_noblur_NoequalizeHist.16-0.99611.hdf5')

names = []
for id in ids_train:
    names.append('{}.jpg'.format(id))


train_splits = len(ids_train)  # Split test set (number of splits must be multiple of 2)
ids_train_splits = np.array_split(ids_train, indices_or_sections=train_splits)

split_count = 0
for ids_train_split in ids_train_splits:
    split_count += 1

    def train_generator():
        while True:
            for start in range(0, len(ids_train_split), batch_size):
                x_batch = []
                end = min(start + batch_size, len(ids_train_split))
                ids_train_split_batch = ids_train_split[start:end]
                for id in ids_train_split_batch.values:
                    img = cv2.imread('input/test_hq/{}.jpg'.format(id))
                    img = cv2.resize(img, (input_w, input_h))
                    #img = cv2.bilateralFilter(img, 3, 40, 40)
                    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (input_w, input_h))
                    mask = np.expand_dims(mask, axis=2)
                    x_batch.append(img)
                    y_batch.append(mask)
                    x_batch = np.array(x_batch, np.float32) / 255
                    y_batch = np.array(y_batch, np.float32) / 255
                yield x_batch


    print("Predicting on {} samples (split {}/{})".format(len(ids_train_split), split_count, train_splits))
    preds = model.predict_generator(generator=train_generator(),
                                    steps=(len(ids_train_split) // batch_size) + 1, verbose=1)

    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (input_w, input_h))
    mask = np.expand_dims(mask, axis=2)

    dice_coeff = dice_coeff(mask, preds)
    print('dice_coeff={} image name:{}'.format(dice_coeff,ids_train_split))
