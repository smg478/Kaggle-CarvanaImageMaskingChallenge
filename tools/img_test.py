import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from keras.losses import binary_crossentropy
import keras.backend as K


from u_net import get_unet_1024_8

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score


model1 = get_unet_1024_8()
#model2 = get_unet_640_960_8()
model1.load_weights(filepath='weights/unet_8_1024_rand1042_ShiftFlipHue_hardExmpl_testSplit20.04-0.99651-0.99643.hdf5')
#model1.load_weights(filepath='weights/unet_8_960_topHalf_ShiftFlipHue_hardExmpl_testSplit20.19-0.99665-0.99656.hdf5')
#model2.load_weights(filepath='weights/unet_8_960_bottomHalf_ShiftFlipHue_hardExmpl_testSplit20.15-0.99607-0.99609.hdf5')


df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

results = []
sum_dice = 0
for id in ids_train:

    img = cv2.imread('/home/galib/kaggle/car_segment/carvana-keras/input/train_hq/{}.jpg'.format(id))
    img = cv2.resize(img, (1024, 1024))
    img = img.astype('float32')
    img = img / 255
    img = img.reshape((1, 1024, 1024, 3))

    #img1 = img[:320, :]
    #img1 = img1.astype('float32')
    #img1 = img1 / 255
    #img1 = img1.reshape((1, 320, 960, 3))
    #
    #img2 = img[320:, :]
    #img2 = img2.astype('float32')
    #img2 = img2 / 255
    #img2 = img2.reshape((1, 320, 960, 3))

    pred = model1.predict(img, verbose=0, batch_size=1)
    pred = np.squeeze(pred)
    #
    #pred1 = model1.predict(img1, verbose=0, batch_size=1)
    #pred1 = np.squeeze(pred1)
    #
    #pred2 = model2.predict(img2, verbose=0, batch_size=1)
    #pred2 = np.squeeze(pred2)

    #pred = np.concatenate((pred1,pred2),axis=0)


    mask_pred = pred * 255
    mask_pred = mask_pred.astype('float32')
    mask_pred[mask_pred < 128] = 0
    cv2.imshow("mask", mask_pred)
    cv2.waitKey(0)
    #

    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (960, 640 ))
    #mask = mask[320:, :]
    #mask = mask[:320, :]
    mask = np.array(mask, np.float32) / 255
    mask = np.expand_dims(mask, axis=2)
    dice_coefff = []

    dice_coefff = dice_coeff(mask, pred)
    print('dice_coeff={} image name:{}'.format(dice_coefff,id))

    results.append(dice_coefff)
    sum_dice = sum_dice+dice_coefff
    avg_dice = sum_dice / len(results)
    print('avg_dice_coeff={}'.format(avg_dice))

    #mask = pred*255
    #mask = mask.astype('float32')
    #mask1 = mask.reshape((640, 960, 1))
    #mask1[mask1<128] = 0

    #cv2.imshow("mask1", mask1)

#np.savetxt("track_dice.csv", results, delimiter=",")

'''
a = np.array([[1,5,9],[2,6,10]])
b = np.array([[3,7,11],[4,8,12]])
#concatenates along the 1st axis (rows of the 1st, then rows of the 2nd):

print concatenate((a,b),axis=0)
array([[ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11],
       [ 4,  8, 12]])
'''