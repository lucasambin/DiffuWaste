import json
import cv2
import os

import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import glob

def retrive_image_mask_object(img_name, mask_name):
    img = cv2.imread(img_name)
    msk = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i,j] != 0:
                msk[i, j] = 255
            else:
                img[i, j] = [0, 0, 0]

    return img, msk

if __name__ == '__main__':

    # Load Image and Mask
    input_path = "path"
    images_path = sorted(glob.glob1('path/img/', '*'))
    masks_path = sorted(glob.glob1('path/label/', '*'))

    # Create a folder to save the images
    # 1-> rigid_plastic, 3 -> metal, 4 -> soft plastic
    images_outpath = "path/images/"
    masks_outpath = "path/masks/"

    print("number of mask: ", len(masks_path))
    print(images_path[0])
    print(masks_path[0])
    count = 0

    # Get a single annotation
    for frame_name in masks_path:
        if count % 10 == 0:
            print("Prov: ", count)

        img_in_path = input_path + "img/" + frame_name
        msk_in_path = input_path + "label/" + frame_name

        # Retrieve the mask of the image and extract the single object patch
        out_img, out_msk = retrive_image_mask_object(img_in_path, msk_in_path)
        img_out_path = images_outpath + frame_name
        msk_out_path = masks_outpath + frame_name
        cv2.imwrite(img_out_path, out_img)
        cv2.imwrite(msk_out_path, out_msk)
        count +=1

    print("patches extracted: ", count)