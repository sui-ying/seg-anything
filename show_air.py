import os
import random
import cv2
import numpy

ori_img = cv2.imread("/cv/xyc/segment-anything/notebooks/images/car.jpg")

mask_img = numpy.zeros_like(ori_img)
mask_path = "/cv/xyc/segment-anything/output/car1"
all_mask_img = os.listdir(mask_path)
random.shuffle(all_mask_img)

for label in all_mask_img:
    if os.path.splitext(label)[-1] == ".png":
        single_mask_img = cv2.imread(os.path.join(mask_path, label),-1)
        index = numpy.argwhere(single_mask_img==255)
        color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        for single_index in index:
            x,y = single_index[0], single_index[1]
            mask_img[x,y,:] = color
# cv2.imshow("mask", mask_img)
# cv2.waitKey()

cv2.imwrite("./dog.jpg", mask_img)
