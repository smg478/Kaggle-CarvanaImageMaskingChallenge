import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

# input_size = 512

input_h = 640  # 1280  #640
input_w = 960  # 1920  #960
batch_size = 8

orig_width = 1918
orig_height = 1280

threshold = 0.5

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))



test_splits = 64  # Split test set (number of splits must be multiple of 2)
ids_test_splits = np.array_split(ids_test, indices_or_sections=test_splits)

split_count = 0
for ids_test_split in ids_test_splits:
    split_count += 1

    x1 = np.load('/submit/probs-part%d.7.npy' % split_count)  # 1min
    x2 = np.load('/submit/probs-part%d.8.npy' % split_count)  # 1min
    preds = np.array(x1+x2)/2


    print("Generating masks...")
    for pred in tqdm(preds, miniters=1000):
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit_unet-960_8_Shift_Flip_Hue_testSplit20_06-0.99615-0.99648.csv.gz', index=False, compression='gzip')