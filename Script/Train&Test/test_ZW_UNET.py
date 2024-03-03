import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
from torch.autograd import Variable

from PIL import Image
import cv2

import time
import os
from tqdm import tqdm

from utils.dataset import SemSegDataset, FilteredSemSegDataset
import albumentations as A

import segmentation_models_pytorch as smp
from utils.metrics import pixel_accuracy, mIoU, get_lr
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils.util import save_tensorboard_images, get_images_with_mask, normalizeRGB
from torchvision.utils import draw_segmentation_masks
from torchmetrics import JaccardIndex


# --------------------------------------------------------------------------------
# Define evaluation metrics
# --------------------------------------------------------------------------------
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), iu.cpu().numpy().astype(float), acc.item()

    def get_matrix(self):
        return self.mat

    def plot_confusion_matrix(self, class_names, save_path=None):
        cm = self.mat.cpu().numpy()
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        figure = plt.figure(figsize=(5.5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # # Normalize the confusion matrix.
        # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close('all')


def predict(model, image, mask, device, n_classes):
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        output = model(image)
        score = mIoU(output, mask, n_classes=n_classes)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def inference(model, image, device, n_classes):
    model.eval()
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    
    #color = ['black', 'yellow']
    color = ['black', 'yellow', 'red', 'green']
    masked_images = get_images_with_mask(images=image, masks_logits=output, color=color)
    
    return output, masked_images

def main():

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--dataset', type=str, default="")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # test_img_folder = "./dataset/img"
    # test_mask_folder = "./dataset/label"
    # test_res_folder = "./dataset/res"
    test_img_folder = "/mnt/Dataset/Test_Single_Patch/ZeroWaste/Images"
    test_mask_folder = "/mnt/Dataset/Test_Single_Patch/ZeroWaste/Masks"
    test_res_folder = "/mnt/Dataset/Test_Single_Patch/ZeroWaste/Results_SYN"

    # Assuming you have the test_res_folder variable defined with the path to your desired folder
    output_folder = os.path.join(test_res_folder, args.dataset)  # Create a subfolder called 'images' inside test_res_folder

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the output folder and subfolders if they don't exist
    os.makedirs(os.path.join(output_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'probability_map'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'pred_mask'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'gt_mask'), exist_ok=True)

    n_classes = 4

    # CDC
    
    t_test = A.Compose([
                        A.Resize(256, 512, p=1.0),
                        A.CenterCrop(256, 256, p=1.0),
                        A.Resize(608, 608, p=1.0)
                      ])
    
                          
    # PBE
    '''
    t_test = A.Compose([
                        A.Resize(512, 512, p=1.0),
                        A.CenterCrop(512, 512, p=1.0),
                        A.Resize(608, 608, p=1.0)
                      ])
    '''

    # BASELINE
    '''
    t_test = A.Compose([
                        A.Resize(1080, 1920, p=1.0),
                        A.CenterCrop(1080, 1080, p=1.0),
                        A.Resize(608, 608, p=1.0)
                      ])
    '''

    test_set = SemSegDataset(test_img_folder, test_mask_folder, transform=t_test)

    # dataloader
    batch_size = 1
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


    # Main 3 models used also in the experiments section of thesis
    #define segmentation model
    model = smp.Unet('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('tu-mobilevitv2_200', encoder_weights="imagenet", classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('mobilenet_v2', encoder_weights="imagenet", classes=n_classes, activation=None)
    
   
   
    
    #model = smp.Unet('resnet50', encoder_weights="imagenet", classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None)
    model.load_state_dict(torch.load('/mnt/saved_models/' + args.dataset + '/best_miou.pth')) #  tpr,  equilibrium
    model.eval()

    confusion_matrix = ConfMatrix(n_classes)

    for i,data in tqdm(enumerate((test_loader))):

        image, gt_mask = data

        logits, pred_masked_image = inference(model, image, device, n_classes)
        pred_mask =  torch.argmax(logits.detach(), dim=1).unsqueeze(1)
        logits = torch.nn.functional.softmax(logits, dim=1)
        image = normalizeRGB(images=image, use_int8=True)
        prob_img = image.to(device) * logits[:,1,:,:].repeat(1,3,1,1)
        prob_img = prob_img.to(torch.uint8)

        gt_masks_2 = gt_mask > 0
        gt_masked_image = draw_segmentation_masks(image[0], masks=gt_masks_2, alpha=0.5, colors="red").unsqueeze(0)

        prob_img = prob_img[0].detach().permute(1,2,0).cpu().numpy()
        pred_masked_image = pred_masked_image[0].detach().permute(1,2,0).cpu().numpy()
        gt_masked_image = gt_masked_image[0].detach().permute(1,2,0).cpu().numpy()
        image = image[0].detach().permute(1,2,0).cpu().numpy()
        gt_mask_a = gt_mask[0].cpu().numpy()

        plt.figure("images")
        
        ax = plt.subplot(2,2,1)
        plt.imshow(image)
        ax.set_title("Image")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2,2,2)
        plt.imshow(prob_img)
        ax.set_title("Probability Map")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2,2,3)
        plt.imshow(pred_masked_image)
        ax.set_title("Pred Mask")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2,2,4)
        plt.imshow(gt_mask_a)
        ax.set_title("GT Mask")
        ax.set_xticks([])
        ax.set_yticks([])

        #METRICS

        # compute metrics by confusion matrix
        gt_mask = gt_mask.cuda()
        confusion_matrix.update(pred_mask.flatten(), gt_mask.flatten())

        plt.tight_layout()

        # Save the images as PNG files inside subfolders
        cv2.imwrite(os.path.join(output_folder, 'image', f'image{i}.png'), image)
        cv2.imwrite(os.path.join(output_folder, 'probability_map', f'probability_map{i}.png'), prob_img)
        cv2.imwrite(os.path.join(output_folder, 'pred_mask', f'pred_mask{i}.png'), pred_masked_image)
        cv2.imwrite(os.path.join(output_folder, 'gt_mask', f'gt_mask{i}.png'), gt_mask_a)

        #save images in a single composition
        #plt.savefig(os.path.join(test_res_folder, f"{i}_res.png"), dpi=200)
        #plt.show()
        plt.close()


    print("*************** RESULTS ******************")
    print("mIOU: ", confusion_matrix.get_metrics()[0])
    print("IOU per class: ", confusion_matrix.get_metrics()[1])
    print("******************************************")



if __name__ == "__main__":
    main()


