import numpy as np
import matplotlib.pyplot as plt

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

from utils.dataset import SemSegDataset
from utils.util import save_tensorboard_images, get_images_with_mask

import segmentation_models_pytorch as smp
from utils.metrics import pixel_accuracy, mIoU, get_lr
from torch.utils.tensorboard import SummaryWriter
import argparse
import albumentations as A

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, writer, device, n_classes, patch=False, args=None):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    max_miou = 0
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data

            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask, n_classes=n_classes)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            lr = get_lr(optimizer)
            scheduler.step()

            running_loss += loss.item()

        
        #SAVE IMAGES
        #print("Output: ", output.shape)
        # color = ['black', 'yellow']
        color = ['black', 'yellow', 'red', 'green']
        masked_images = get_images_with_mask(images=image, masks_logits=output, color=color)
        save_tensorboard_images(images=masked_images, label="[TRAIN OUT]", logger=writer, iters = e)
        mask = mask.unsqueeze(1)
        mask = torch.cat([1-mask, mask], dim=1)
        #print("mask: ", mask.shape)
        #print("mask: ", torch.unique)
        masked_images = get_images_with_mask(images=image, masks_logits=mask.float())
        save_tensorboard_images(images=masked_images, label="[TRAIN GT]", logger=writer, iters = e, color=color)

 #       else:
        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0
        # validation loop
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                # reshape to 9 patches from single image, delete batch size
                image_tiles, mask_tiles = data

                if patch:
                    bs, n_tiles, c, h, w = image_tiles.size()

                    image_tiles = image_tiles.view(-1, c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)

                image = image_tiles.to(device)
                mask = mask_tiles.to(device)
                output = model(image)
                # evaluation metrics
                val_iou_score += mIoU(output, mask, n_classes=n_classes)
                test_accuracy += pixel_accuracy(output, mask)
                # loss
                loss = criterion(output, mask)
                test_loss += loss.item()

        # calculation mean error for each batch
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(val_loader))

        writer.add_scalars("Loss / Epoch", {'Train': running_loss / len(train_loader), 'Val': test_loss / len(val_loader) }, e)
        writer.add_scalar("LR / Epoch", lr, e)

        if (val_iou_score / len(val_loader)) > max_miou:
            max_miou = val_iou_score / len(val_loader)
            print('update best model...')
            torch.save(model.state_dict(), os.path.join('/mnt/saved_models', args.train_id, 'best_miou.pth'))

        if (test_loss / len(val_loader)) < min_loss:
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
            min_loss = (test_loss / len(val_loader))
            decrease += 1
            not_improve = 0 #ADD TO RESET THE COUNT
            if decrease % 5 == 0 or e % 5 == 0:
                print('saving model...')
                torch.save(model.state_dict(), os.path.join('/mnt/saved_models', args.train_id, 'ep_{}_mIoU-{:.3f}.pth'.format(e, val_iou_score / len(val_loader))))
        else:
            not_improve += 1
            #min_loss = (test_loss / len(val_loader))
            print(f'Loss Not Decrease for {not_improve} time')
            if not_improve == args.overfit_epoch:
                print('Loss not decrease for {} times, Stop Training'.format(args.overfit_epoch))
                break

        # iou
        val_iou.append(val_iou_score / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        train_acc.append(accuracy / len(train_loader))
        val_acc.append(test_accuracy / len(val_loader))

        writer.add_scalars("Accuracy / Epoch", {'Train': accuracy / len(train_loader), 'Val': test_accuracy / len(val_loader) }, e)
        writer.add_scalars("mIoU / Epoch", {'Train': iou_score / len(train_loader), 'Val': val_iou_score / len(val_loader)}, e)

        #SAVE IMAGES
        #print("Output: ", output.shape)
        masked_images = get_images_with_mask(images=image, masks_logits=output, color=color)
        save_tensorboard_images(images=masked_images, label="[VAL OUT]", logger=writer, iters = e)

        mask = mask.unsqueeze(1)
        mask = torch.cat([1-mask, mask], dim=1)
        #print("mask: ", mask.shape)
        #print("mask: ", torch.unique)
        masked_images = get_images_with_mask(images=image, masks_logits=mask.float())
        save_tensorboard_images(images=masked_images, label="[VAL GT]", logger=writer, iters = e, color=color)
        
        print("Epoch:{}/{}..".format(e + 1, epochs),
              "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
              "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
              "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
              "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
              "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
              "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
              "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def main():

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--train_id', type=str, default="")
    parser.add_argument('--overfit_epoch', type=int, default=5)
    parser.add_argument('--dataset', type=str, default="")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join("/mnt/runs", args.train_id))

    root = "/mnt"
    dataset = 'Dataset/Synthetic/ZW/'

    train_img_folder = os.path.join(root, dataset, args.dataset + "/Images")
    train_mask_folder = os.path.join(root, dataset, args.dataset + "/Masks")

    val_img_folder = os.path.join(root, dataset, "/mnt/Dataset/Single_Patch/ZeroWaste/Validation/Images")
    val_mask_folder = os.path.join(root, dataset, "/mnt/Dataset/Single_Patch/ZeroWaste/Validation/Masks")

    save_folder = os.path.join('/mnt/saved_models', args.train_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    n_classes = 4

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    t_train = A.Compose([A.Resize(256, 512, interpolation=cv2.INTER_LINEAR), A.CenterCrop(256, 256), A.Resize(608, 608, interpolation=cv2.INTER_LINEAR), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                        A.GridDistortion(p=0.25), A.RandomRotate90(p=0.5), A.Transpose(p=0.5), A.RandomBrightnessContrast((0,0.5),(0,0.5),p=0.2),
                        A.GaussNoise(p=0.5)])

    t_val = A.Compose([A.Resize(256, 512, interpolation=cv2.INTER_LINEAR), A.CenterCrop(256,256), A.Resize(608, 608, interpolation=cv2.INTER_LINEAR), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                    A.GridDistortion(p=0.2)])

    # t_train = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    #                    A.RandomRotate90(p=0.5), A.Transpose(p=0.5), A.RandomBrightnessContrast((0,0.5),(0,0.5),p=0.2),
    #                    A.GaussNoise(p=0.5)])
    #
    # t_val = A.Compose([A.VerticalFlip(p=0.5)])

    train_set = SemSegDataset(train_img_folder, train_mask_folder, transform=t_train)
    val_set = SemSegDataset(val_img_folder, val_mask_folder, transform=t_val)

    # dataloader
    batch_size = args.batch

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Main 3 models used also in the experiments section of thesis
    model = smp.Unet('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('tu-mobilevitv2_200', encoder_weights="imagenet", classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('mobilenet_v2', encoder_weights="imagenet", classes=n_classes, activation=None)





    #model = smp.Unet('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.Unet('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[512, 256, 128, 64, 32])
    
    #model = smp.UnetPlusPlus('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    
    #model = smp.DeepLabV3Plus('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('resnet101', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('timm-mobilenetv3_large_100', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.Unet('mit_b2', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.DeepLabV3Plus('tu-tf_mixnet_l', encoder_weights='imagenet', classes=n_classes, activation=None)

    #model = smp.FPN('resnet50', encoder_weights="imagenet", classes=n_classes, activation=None)
    #model = smp.Unet('tu-mobilevitv2_150', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.Unet('resnet101', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[512, 256, 128, 64, 32])

    #model = smp.MANet('resnet50', encoder_weights='imagenet', classes=n_classes, activation=None)


    #FREEZE BACKBONE LAYERS
    #for count, child in enumerate(model.children()):
    #     print(child)

    # for count, child in enumerate(model.children()):
    #     if count == 0:
    #             for param in child.parameters():
    #                 param.requires_grad = False

    # for count, child in enumerate(model.children()):
    #     if count == 0:
    #         for count2, child2 in enumerate(child.children()):
    #             if count2 < 18: #max 18
    #                 for param in child2.parameters():
    #                     param.requires_grad = False

    max_lr = 1.0e-4
    epoch = 50
    weight_decay = 1.0e-5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

    history = fit(epochs= epoch,model= model,train_loader= train_loader,val_loader= val_loader,criterion= criterion,optimizer= optimizer, scheduler= sched,writer= writer,device= device,n_classes= n_classes, args= args)


if __name__ == "__main__":
    main()


