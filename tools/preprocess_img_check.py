import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import array, arange

def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def contrast_enhance(img,phi = 1, theta = 1):
    maxIntensity = 255.0 # depends on dtype of image data
    newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**1.0
    return np.uint8(newImage0)

def equalize_hist(img):
    for c in range(0, 2):
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #img[:,:,c] = clahe.apply(img[:,:,c])
        img[:,:,c] = cv2.equalizeHist(img[:,:,c])
    return np.uint8(img)

img_org = cv2.imread('/home/galib/kaggle/car_segment/carvana-keras/input/train/0de66245f268_14.jpg') #gray image
#img_org = np.transpose(img_org, (0, 2, 3, 1))

#dst = contrast_enhance(img,1, 1)

#equ = equalize_hist(img_org)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#equ = clahe.apply(img_org)

gamma = gamma_correction(img_org, 1.3)

#blur = cv2.blur(dst,(3,3))
blur = cv2.bilateralFilter(img_org,3,40,40)

plt.subplot(221),plt.imshow(img_org),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(gamma),plt.title('gamma')
plt.xticks([]), plt.yticks([])
#plt.subplot(223),plt.imshow(gamma),plt.title('gamma')
#plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imshow("gamma corrected", dst)

#cv2.waitKey(0)
