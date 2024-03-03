import random
import cv2
import numpy as np
import glob
from random import choices

def copy_paste(back_img, back_msk_1, patch_name, patch_mask_name_1):
    patch = cv2.imread(patch_name)
    msk_1 = cv2.imread(patch_mask_name_1, cv2.IMREAD_GRAYSCALE)
    out_img = back_img
    out_msk_1 = back_msk_1

    # Set threshold level
    threshold_level = 0

    # Find coordinates of all pixels different from threshold
    # Create mask of all pixels lower than threshold level
    maskered_1 = msk_1 != threshold_level

    # Extract random coordinate
    # Width
    x = random.randint(0, out_img.shape[1] - msk_1.shape[1])
    # Height
    y = random.randint(0, out_img.shape[0] - msk_1.shape[0])

    # Change only the selected ROI
    tmp_img = out_img[y:y+msk_1.shape[0], x:x+msk_1.shape[1]]
    tmp_msk_1 = out_msk_1[y:y + msk_1.shape[0], x:x + msk_1.shape[1]]

    # Color the pixels in the mask
    tmp_msk_1[maskered_1] = msk_1[maskered_1]
    tmp_img[maskered_1] = patch[maskered_1]
    return out_img, out_msk_1

if __name__ == '__main__':
    # Load Background, Images and Masks
    background_path = sorted(glob.glob("path/Background/Images/*"))
    rigid_images_path = "path/Patch/Images/rigid_plastic/"
    rigid_mask_path_1 = "path/Patch/Masks/rigid_plastic/"
    soft_images_path = "path/Patch/Images/soft_plastic/"
    soft_mask_path_1 = "path/Patch/Masks/soft_plastic/"
    metal_images_path = "path/Patch/Images/metal/"
    metal_mask_path_1 = "path/Patch/Masks/metal/"

    rigid_patch_name = sorted(glob.glob1(rigid_images_path, '*'))
    soft_patch_name = sorted(glob.glob1(soft_images_path, '*'))
    metal_patch_name = sorted(glob.glob1(metal_images_path, '*'))

    # Create a folder to save the images
    images_outpath = "path/output/Images/"
    masks_outpath_255 = "path/output/Masks/"
    masks_outpath_1 = "path/output/Masks_123/"

    size = len(rigid_patch_name) + len(soft_patch_name) + len(metal_patch_name)
    print("number of patches: ", size)
    print("number of background: ", len(background_path))

    # Class balance - same probability for each class
    population = [1, 2, 3]
    weights = [1 / 3, 1 / 3, 1 / 3]

    # Set seed for data repetition
    random.seed(100)

    # Set number of required images
    count = 0
    l = 1
    to_be_generated = 3300

    while count < to_be_generated:
        # Print advancement
        if count % 100 == 0:
            print("Percentual: ", count / to_be_generated * 100)

        # Get a random background
        i_back = random.randint(0, len(background_path)-1)

        # Get a random number of patches to be composed
        # MAX 4 patches for ZeroWaste Dataset
        num_patches = random.randint(1, 4)

        tmp_img = cv2.imread(background_path[i_back])
        tmp_msk_1 = np.zeros((tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8)
        # Iterate until all the patches are copy&paste
        while num_patches > 0:
            # Class Balance
            a = choices(population, weights)
            ith_patch_idx = a[0]

            # rigid plastic
            if ith_patch_idx == 1:
                idx = random.randint(0, len(rigid_patch_name) - 1)
                img_in_path = rigid_images_path + rigid_patch_name[idx]
                msk_in_path_1 = rigid_mask_path_1 + rigid_patch_name[idx]

            # soft plastic
            elif ith_patch_idx == 2:
                idx = random.randint(0, len(soft_patch_name) - 1)
                img_in_path = soft_images_path + soft_patch_name[idx]
                msk_in_path_1 = soft_mask_path_1 + soft_patch_name[idx]

            # metal
            elif ith_patch_idx == 3:
                idx = random.randint(0, len(metal_patch_name) - 1)
                img_in_path = metal_images_path + metal_patch_name[idx]
                msk_in_path_1 = metal_mask_path_1 + metal_patch_name[idx]

            else:
                print("WRONG")
                exit(-1234)

            # Copy and Paste the patch in the background
            out_img, out_mask_1 = copy_paste(tmp_img, tmp_msk_1, img_in_path, msk_in_path_1)

            # Save Temporary Image and Mask
            tmp_img = out_img
            tmp_msk_1 = out_mask_1

            # Update Variable
            num_patches -= 1

        # Get 5 digit string from number
        s = "{0:05}".format(l)

        # Save the final composition
        # 100%
        if count < 600:
            img_out_path = images_outpath + "ZW_Train_" + s + "_1.PNG"
            msk_out_path_255 = masks_outpath_255 + "ZW_Train_" + s + "_1.PNG"
            msk_out_path_1 = masks_outpath_1 + "ZW_Train_" + s + "_1.PNG"

        # 500%
        elif count < 1800:
            img_out_path = images_outpath + "ZW_Train_" + s + "_2.PNG"
            msk_out_path_255 = masks_outpath_255 + "ZW_Train_" + s + "_2.PNG"
            msk_out_path_1 = masks_outpath_1 + "ZW_Train_" + s + "_2.PNG"

        # 1000%
        else:
            img_out_path = images_outpath + "ZW_Train_" + s + "_3.PNG"
            msk_out_path_255 = masks_outpath_255 + "ZW_Train_" + s + "_3.PNG"
            msk_out_path_1 = masks_outpath_1 + "ZW_Train_" + s + "_3.PNG"

        # Save images
        tmp_msk_255 = np.zeros((tmp_msk_1.shape[0], tmp_msk_1.shape[1]), dtype=np.uint8)
        tmp_msk_255[tmp_msk_1 != 0] = 255
        cv2.imwrite(img_out_path, tmp_img)
        cv2.imwrite(msk_out_path_255, tmp_msk_255)
        cv2.imwrite(msk_out_path_1, tmp_msk_1)
        count += 1
        l += 1

    print("Number of file generates: ", count)
    print("Found Errors? ", count != size)