import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm


df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

orig_width = 1918
orig_height = 1280

threshold = 0.5

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):

    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

rles = []

test_splits = 64  # Split test set (number of splits must be multiple of 2)
ids_test_splits = np.array_split(ids_test, indices_or_sections=test_splits)

split_count = 4
for ids_test_split in ids_test_splits:
    split_count += 1
    print(split_count)
    preds1 = []
    preds2 = []
    preds = []
    with open('/media/galib/Documents/carvana/probs-part%d.u8.npy' % split_count , "rb") as npy1:
        preds1 = np.load(npy1)
    with open('/media/galib/Documents/carvana/probs-part%d.u11.npy' % split_count, "rb") as npy2:
        preds2 = np.load(npy2)
    #preds1 = np.load('/media/galib/Documents/carvana/probs-part%d.u8.npy' % split_count )  # 1min
    print('loaded preds1(%d)...'%split_count)
    print(preds1.shape)
    #preds2 = np.load('/media/galib/Documents/carvana/probs-part%d.u11.npy' % split_count)  # 1min
    print('loaded preds2(%d)...'%split_count)
    print(preds2.shape)


    preds = np.array((preds1*0.65)+(preds2*0.35))
    print(preds.shape)
    #preds = preds/2
    del preds1, preds2

    print("Generating masks...")
    for pred in tqdm(preds, miniters=100):
        prob = cv2.resize (pred, (orig_width, orig_height))
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/average_unet-960_8_11_3.csv.gz', index=False, compression='gzip')