## Kaggle Carvana Image Masking Challange 2017 solution on keras.

This repository contains the keras solution files of the challange. \\
The file 'u_net_models.py' contains definitions of one basic and two modified u-nets. 
Features of modified u-net:
Resudual blocks and inception blocks.
Skip connections.
High resolution training.
## Disclaimer
This solution is based on Peter's starter code for the competition. Thanks to him!  
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

Train \\

Run python train.py to train the model. \\ 

Test and submit

Run python test_submit.py to make predictions on test data and generate submission.


