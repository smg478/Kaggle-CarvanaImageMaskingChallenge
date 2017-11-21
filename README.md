## Kaggle Carvana Image Masking Challange 2017 solution on keras

This repository contains the keras solution files of the challange. 
Results produced from this algorithm ranks below 50 in the private leaderboard of the competition.

Usage of the code is mostly similar to Peter's keras implementation of the solution. Thanks to him for nice starter code!

The file 'u_net_models.py' contains definitions of one basic and two modified u-nets. 

## Features of modified u-net

Resudual blocks and inception blocks

Skip connections

High resolution training

You can experiment with your own idea also!


## Requirements

Keras 2.0 w/ TF backend
sklearn
cv2
tqdm
h5py

## Usage

### Data

Place 'train', 'train_masks' and 'test' data folders in the 'input' folder.

Convert training masks to .png format. You can do this with:

mogrify -format png *.gif

in the 'train_masks' data folder.

### Train

Run python train.py to train the model. 

### Test and submit

Run python test_submit_multithreaded.py to make predictions on test data and generate submission.

Run python test_submit_ensemble.py to make weighted average ensemble from submission files.


