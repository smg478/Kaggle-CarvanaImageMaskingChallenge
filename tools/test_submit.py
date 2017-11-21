import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from u_net import get_unet_640_960_11


df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

input_h = 640 #1280  #640
input_w = 960 #1920  #960
batch_size = 8

orig_width = 1918
orig_height = 1280

threshold = 0.5

model = get_unet_640_960_11()
model.load_weights(filepath='weights/unet_11_960_ShiftFlipHue_hardExmpl_testSplit20.07-0.99652-0.99641.hdf5')

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    inds = mask.flatten()
    inds[0] = 0
    inds[-1] = 0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

rles = []

test_splits = 64  # Split test set (number of splits must be multiple of 2)
ids_test_splits = np.array_split(ids_test, indices_or_sections=test_splits)

split_count = 0
for ids_test_split in ids_test_splits:
    split_count += 1

    def test_generator():
        while True:
            for start in range(0, len(ids_test_split), batch_size):
                x_batch = []
                end = min(start + batch_size, len(ids_test_split))
                ids_test_split_batch = ids_test_split[start:end]
                for id in ids_test_split_batch.values:
                    img = cv2.imread('input/test_hq/{}.jpg'.format(id))
                    img = cv2.resize(img, (input_w, input_h))
                    #img = cv2.bilateralFilter(img, 3, 40, 40)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32) / 255
                yield x_batch


    print("Predicting on {} samples (split {}/{})".format(len(ids_test_split), split_count, test_splits))

    preds = model.predict_generator(generator=test_generator(),
                                    steps=(len(ids_test_split) // batch_size) + 1, verbose=1)

    preds = np.array(preds)

    print("making numpy array ...")
    preds = np.squeeze(preds, axis=3)

    print("Generating masks...")
    for pred in tqdm(preds, miniters=1000):
        prob = cv2.resize(pred, (orig_width, orig_height))
        #
        #mask = prob * 255
        #mask = mask.astype('float32')
        #mask[mask < 128] = 0
        #cv2.imshow("mask", mask)
        #
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/unet_11_960_ShiftFlipHue_hardExmpl_testSplit20.07-0.99652-0.99641.csv.gz', index=False, compression='gzip')