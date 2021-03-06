# Thanks to Peter Giannakopoulos https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523

import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from tqdm import tqdm
from u_net import get_unet1_1024x1024


input_h = 1024
input_w = 1024
batch_size = 4

orig_width = 1918
orig_height = 1280

threshold = 0.5

model = get_unet1_1024x1024()
model.load_weights(filepath='weights/unet1_05-0.99647-0.99634.hdf5')

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
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


rles = []

graph = tf.get_default_graph()

q_size = 10


def data_loader(q, ):
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            img = cv2.imread('input/test_hq/{}.jpg'.format(id))
            img = cv2.resize(img, (input_w, input_h))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put(x_batch)


def predictor(q, ):
    for i in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = q.get()
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for pred in preds:
            prob = cv2.resize(pred, (orig_width, orig_height))
            mask = prob > threshold
            rle = run_length_encode(mask)
            rles.append(rle)


q = queue.Queue(maxsize=q_size)
t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))
print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
t1.start()
t2.start()
# Wait for both threads to finish
t1.join()
t2.join()

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/unet1_05-0.99647-0.99634.csv.gz', index=False, compression='gzip')
