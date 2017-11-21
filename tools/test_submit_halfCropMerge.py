import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from u_net import get_unet_640_960_8


df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

input_h = 640 #1280  #640
input_w = 960 #1920  #960

orig_width = 1918
orig_height = 1280

threshold = 0.5

model1 = get_unet_640_960_8()
model2 = get_unet_640_960_8()
#model1.load_weights(filepath='weights/unet_08_960__ShiftFlipHue_hardExmpl_testSplit20.06-0.99643-0.99631.hdf5')
model1.load_weights(filepath='weights/unet_8_960_topHalf_ShiftFlipHue_hardExmpl_testSplit20.19-0.99665-0.99656.hdf5')
model2.load_weights(filepath='weights/unet_8_960_bottomHalf_ShiftFlipHue_hardExmpl_testSplit20.15-0.99607-0.99609.hdf5')

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


def run_length_encode(mask):
    inds = mask.flatten()
    inds[0] = 0
    inds[-1] = 0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

rles = []

for id in tqdm(ids_test, miniters=100):
    img = cv2.imread('input/test_hq/{}.jpg'.format(id))
    img = cv2.resize(img, (960, 640))

    img1 = img[:320, :]
    img1 = img1.astype('float32')
    img1 = img1 / 255
    img1 = img1.reshape((1, 320, 960, 3))

    img2 = img[320:, :]
    img2 = img2.astype('float32')
    img2 = img2 / 255
    img2 = img2.reshape((1, 320, 960, 3))

    pred1 = model1.predict(img1, verbose=0, batch_size=1)
    pred1 = np.squeeze(pred1)

    pred2 = model2.predict(img2, verbose=0, batch_size=1)
    pred2 = np.squeeze(pred2)

    pred = np.concatenate((pred1,pred2),axis=0)

    prob = cv2.resize(pred, (orig_width, orig_height))
    mask = prob > threshold
    rle = run_length_encode(mask)
    rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/unet_08_960_half_half.csv.gz', index=False, compression='gzip')
