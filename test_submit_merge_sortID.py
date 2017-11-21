
import numpy as np
import pandas as pd
from tqdm import tqdm


import gzip, csv

def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    inds[0] = 0
    inds[-1] = 0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle
    
    
def run_length_decode(rle):
    """
    Decode a run-length encoding (type string) of a 1D array.
    
    """
    s = rle.split(" ") #()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    n=2455040 #orig_width*orig_height
    mask = np.full(n, 0)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask  
    

#######################################################

#orig_width = 1918 
#orig_height = 1280 


df_test = pd.read_csv('input/sample_submission.csv')
df_test= df_test.sort_values('img', ascending=True)
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))



print("Start loading candidate submissions:")
cand1 = pd.read_csv('submit/best_submit/unet_8_1024_rand1042_ShiftFlipHue_hardExmpl_testSplit20.04-0.99651-0.99643.csv.gz', compression='gzip')
cand1 = cand1.sort_values('img', ascending=True)
cand1 =cand1["rle_mask"]

cand2 = pd.read_csv('submit/best_submit/unet_8_1024x1536_rand1536_FlipHue_hardExmpl_testSplit20.05-0.99657-0.99641.csv.gz', compression='gzip')
cand2 = cand2.sort_values('img', ascending=True)
cand2 =cand2["rle_mask"]

cand3 = pd.read_csv('submit/best_submit/unet_8_832x1216_rand742_FlipHue_hardExmpl_testSplit20.02-0.99649-0.99644.csv.gz', compression='gzip')
cand3 = cand3.sort_values('img', ascending=True)
cand3 =cand3["rle_mask"]

#cand4 = pd.read_csv('submit/unet_11_960_ShiftFlipHue_hardExmpl_testSplit20.07-0.99652-0.99641.csv.gz', compression='gzip')
#cand4 = cand4.sort('img', ascending=True)
#cand4 =cand4["rle_mask"]
print("Done loading candidate submissions:")


##########################################################

rles = []

print('Start Doing Merged Submission Predictions:')
for i in tqdm(range(len(ids_test)), miniters=1000):
    pred_merge=1.0*run_length_decode(cand1[i])+1.0*run_length_decode(cand2[i])+1.0*run_length_decode(cand3[i])
    #pred_merge = 1.0 * run_length_decode(cand1[i]) + 1.0 * run_length_decode(cand2[i])
    pred_merge=pred_merge> 1.9
    rle=run_length_encode(pred_merge)
    rles.append(rle)
    
#######################################################

print("Generating submission file...")

df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/best3_u8_1024x1024_832x1216_1024x1536.csv.gz', index=False, compression='gzip')
print("All done!")


#######################################################
