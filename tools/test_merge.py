import numpy as np
import pandas as pd

import gzip, csv


def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def run_length_decode(rle):
    """
    Decode a run-length encoding (type string) of a 1D array.

    """
    s = rle.split(" ")  # ()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    n = 2455040  # orig_width*orig_height
    mask = np.full(n, 0)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


#######################################################

# orig_width = 1918
# orig_height = 1280


df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))

print("Start loading candidate submissions:")
cand1 = pd.read_csv('submit/submission_07.csv.gz', compression='gzip')
cand1 = cand1["rle_mask"]

cand2 = pd.read_csv('submit/submission_ens01.csv.gz', compression='gzip')
cand2 = cand2["rle_mask"]

cand3 = pd.read_csv('submit/submission_ens01_hq.csv.gz', compression='gzip')
cand3 = cand3["rle_mask"]
print("Done loading candidate submissions:")

##########################################################

rles = []

print('Start Doing Merged Submission Predictions:')
for i in range(0, len(ids_test)):
    pred_merge = 1.0 * run_length_decode(cand1[i]) + 1.0 * run_length_decode(cand2[i]) + 1.0 * run_length_decode(
        cand3[i])
    pred_merge = pred_merge > 1.9
    rle = run_length_encode(pred_merge)
    rles.append(rle)
    if i % 1000 == 0:
        print
        i  # , pred_merge, pred_merge.shape
        # print cand1[i],cand2[i]
# if i<200:
#            print rle
#            print cand1[i]
#            print pred_merge

#######################################################

print("Generating submission file...")

df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission_merge01.csv.gz', index=False, compression='gzip')
print("All done!")


#######################################################