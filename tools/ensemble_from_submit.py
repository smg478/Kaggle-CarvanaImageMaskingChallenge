# Run-Length Encode and Decode

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# Time Test

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img']

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


masks1 = pd.read_csv('submit/unet-960_8_noblur_NoequalizeHist.16-0.99611.csv')
num_masks1 = masks1.shape[0]
print('Total masks1 to encode/decode =', num_masks1)

masks2 = pd.read_csv('submit/unet_08_960__ShiftFlipHue_hardExmpl_testSplit20.06-0.99643-0.99631.csv')
num_masks2 = masks2.shape[0]
print('Total masks2 to encode/decode =', num_masks2)


rles = []
for id in ids_test:
    rle1 = masks1.values.id.rle_mask
    rle2 = masks2.values.id.rle_mask

    mask1 = rle_decode(rle1, (1280, 1918))
    mask2 = rle_decode(rle2, (1280, 1918))

    final_mask = (mask1+mask2) / 2
    mask = final_mask > .51
    rle = rle_encode(mask)
    rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/unet_11_960_ShiftFlipHue_hardExmpl_testSplit20.07-0.99652-0.99641.csv.gz', index=False, compression='gzip')