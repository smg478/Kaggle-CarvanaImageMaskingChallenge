import cv2
img = cv2.imread('input/train_masks/0cdf5b5d0ce1_01_mask.png',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (960, 640))
#crop_img = img[:640, :] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
cv2.imshow("cropped", img)
cv2.waitKey(0)