import random
import cv2
import numpy as np
import glob

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
    images_path = "path/Patch/Images/"
    mask_path_1 = "path/Patch/Masks/"
    patch_name = sorted(glob.glob1(images_path, '*'))

    # Create a folder to save the images
    images_outpath = "path/output/Images/"
    masks_outpath_255 = "path/output/Masks/"
    masks_outpath_1 = "path/output/Masks_123/"

    print("number of patches: ", len(patch_name))
    print("number of background: ", len(background_path))
    # Set seed for data repetition
    random.seed(100)

    # Set number of required images
    count = 0
    l = 1
    i = 0
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
            if i >= len(patch_name):
                i = 0
            img_in_path = images_path + patch_name[i]
            msk_in_path_1 = mask_path_1 + patch_name[i]

            # Copy and Paste the patch in the background
            out_img, out_mask_1 = copy_paste(tmp_img, tmp_msk_1, img_in_path, msk_in_path_1)

            # Save Temporary Image and Mask
            tmp_img = out_img
            tmp_msk_1 = out_mask_1

            # Update Variable
            num_patches -= 1
            i += 1

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
        cv2.imwrite(img_out_path, tmp_img)
        cv2.imwrite(msk_out_path_1, tmp_msk_1)
        tmp_msk_255 = tmp_msk_1
        tmp_msk_255[tmp_msk_255 != 0] = 255
        cv2.imwrite(msk_out_path_255, tmp_msk_255)

        count += 1
        l += 1

    print("Number of file generates: ", count)
    print("Found Errors? ", count != len(patch_name))