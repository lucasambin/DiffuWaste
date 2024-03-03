import json
import cv2
import numpy as np

def retrive_image_mask_object(img_name, mask_name, bbox, category):
    img = cv2.imread(img_name)
    msk = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

    out_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype = np.uint8)
    out_msk = np.zeros((msk.shape[0], msk.shape[1], 1), dtype = np.uint8)

    # bbox: [x1, y1, x2-x1, y2-y1]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    x2 += x1
    y2 += y1
    for i in range(y1, y2, 1):
        for j in range(x1, x2, 1):
            #print (i, j)
            #if msk[i, j] != 0 and msk[i, j] != 2:
            if msk[i, j] == category:
                out_msk[i, j] = category
                out_img[i, j] = img[i, j]

    return out_img[y1:y2, x1:x2], out_msk[y1:y2, x1:x2]

if __name__ == '__main__':

    # Load Image and Mask
    images_path = "path/train/data/"
    masks_path = "path/train/sem_seg/"

    # Create a folder to save the images
    # 1-> rigid_plastic, 3 -> metal, 4 -> soft plastic
    images_outpath = "path/Patch/Img/"
    masks_outpath = "path/Msk/"

    #JSON reader
    with open('path/train/labels.json', 'r') as myfile:
        data=myfile.read()
    obj = json.loads(data)

    print("annotations number: ", len(obj['annotations']))
    count = 0
    prev_img_id = obj['annotations'][0]['image_id']

    # Get a single annotation
    k = 0
    l = 1
    size = len(obj['annotations'])
    while k < size-1:
        if k%100 == 0:
            print("Percentual: ", k/size*100)

        image_id = obj['annotations'][k]['image_id']
        frame_name = obj['images'][image_id]['file_name']

        img_name = images_path + frame_name
        msk_name = masks_path + frame_name

        while image_id == prev_img_id and k < size-1:
            if k % 100 == 0:
                print("Percentual: ", k / size * 100)
            # Now get imageid (order of the image alphabetic), category_id, bbox
            category_id = obj['annotations'][k]['category_id']
            # Save only rigid/soft plastic and metal patches
            if category_id != 2 and category_id != 0:
                bbox = obj['annotations'][k]['bbox']
                # Get 3 digit string from number
                s = "{0:07}".format(l)
                l += 1
                img_out_path = images_outpath + "ZW_" + s + ".PNG"
                msk_out_path = masks_outpath + "ZW_" + s + ".PNG"

                # Retrieve the mask of the image and extract the single object patch
                out_img, out_msk = retrive_image_mask_object(img_name, msk_name, bbox, category_id)
                cv2.imwrite(img_out_path, out_img)
                cv2.imwrite(msk_out_path, out_msk)
                count += 1

            prev_img_id = image_id
            k += 1
            image_id = obj['annotations'][k]['image_id']
        prev_img_id = image_id

    print("patches extracted: ", count)