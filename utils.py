import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from dataset import CarvanaDataset
from PIL import Image
import os
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

PALETTE = ([148, 218, 255], [85, 85, 85], [200, 219, 190], 
           [166, 133, 226], [255, 171, 225])

class_list = [0, 1, 2, 3, 4]

def label_to_color(gt_img):
    """
    Given a ground-truth image, return an rgb image which is easier to interpret visually
    """
    width = gt_img.shape[0]
    height = gt_img.shape[1]
    bgr = np.zeros((width, height, 3), dtype=np.uint8)
    for k, color in enumerate(PALETTE):
        bgr[gt_img == k] = color
    # set background black
    bgr[gt_img == 255] = [0, 0, 0]
    return bgr

def caclulate_dice_nd(class_annotation, class_prediction):
    numer = 2.0 * np.sum(class_annotation & class_prediction)
    denom = np.sum(class_annotation) + np.sum(class_prediction)
    return numer, denom

def calculate_iou_nd(class_annotation, class_prediction):
    numer = np.sum(class_annotation & class_prediction)
    denom = np.sum(class_annotation | class_prediction)
    return numer, denom

def get_class_masks(target, prediction, gt_mask, class_val):
    class_annotation = target == class_val
    class_prediction = (prediction == class_val) & gt_mask
    return class_annotation, class_prediction

def calculate_metrics(target, prediction):
    nd_dict = {"numer": 0, "denom": 0}
    dice_state = {class_val: nd_dict.copy() for class_val in class_list}
    iou_state = {class_val: nd_dict.copy() for class_val in class_list}
    class_metrics = {"dice": dice_state, "iou": iou_state}
    
    gt_mask = np.isin(target, class_list)
    for class_val in class_list:
        class_annotation, class_prediction = get_class_masks(target, prediction, gt_mask, class_val)
        class_metrics["dice"][class_val]["numer"], class_metrics["dice"][class_val]["denom"] = caclulate_dice_nd(class_annotation, class_prediction)
        class_metrics["iou"][class_val]["numer"], class_metrics["iou"][class_val]["denom"] = calculate_iou_nd(class_annotation, class_prediction)
    
    return class_metrics

def calculate_average_metrics(state_dic):
    class_metrics = []
    for class_val in class_list:
        numer = state_dic[class_val]['numer']
        denom = state_dic[class_val]['denom']
        if denom > 0:
            class_metrics.append(numer/denom)
    return float(np.mean(class_metrics))

def get_dice_iou(target, prediction):
    class_metrics = calculate_metrics(target, prediction)
    dice = calculate_average_metrics(class_metrics["dice"])
    iou = calculate_average_metrics(class_metrics["iou"])
    return dice, iou
        
def display_func(display_list, epoch_save_dic=None, epoch=-1):
    plt.figure(figsize=(25, 25))
    
    predicted = ""
    
    if len(display_list) > 2:
        ti = display_list[0].cpu().numpy()
        tp = display_list[1].cpu().numpy()
        dice, iou = get_dice_iou(ti, tp)
        
        predicted = "Predicted Mask DICE = {:.3f}, mIoU = {:.3f}".format(dice, iou)

    title = ["Input Image", 'True Mask', predicted]

    for i in range(len(display_list)):
        plt_image = display_list[i].cpu().numpy()
        plt_image = plt_image.transpose((1, 2, 0)) if i == 0 else label_to_color(plt_image)
        
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(plt_image, cmap='gray')
        plt.axis('off')
    
    if epoch == -1:
        plt.show()
    else:    
        plt.savefig(epoch_save_dic+"epoch_{}.png".format(epoch))

def save_checkpoint(state, filename = "E:\\checkpoint_5mm_3576Im_multi_classes_new_scipt.pth.tar") : 
    print("=> saving checkpoint")
    torch.save(state, filename)
def load_checkpoint(state, model) : 
    print("=> loading checkpoint")
    checkpoint = torch.load(state)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def check_accuracy(loader, model, device = "cuda") : 
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad() : 
        for x, y in loader : 
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.softmax(model(x), dim=1)
            preds = (preds>0.5).float()
            num_correct += (preds == y).sum()
            num_pixels +=torch.numel(preds)
            dice_score += (2*(preds*y).sum())/((preds + y).sum())
    print(f"Got {num_correct}/{num_pixels} with accuracy  {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imags(loader, model, folder="C:\\temp_resultat\\test_results\\predect_images\\", device='cuda', n_classes=8):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        #if n_classes > 1:
            #preds = torch.sum(preds, dim=1)

        # Enregistrement des prédictions
        for i in range(preds.shape[1]):
            pred_channel = preds[:, i]
            torchvision.utils.save_image(pred_channel.unsqueeze(1), f"{folder}/preds_channel_{i}_{idx}.png")

        # Enregistrement des annotations
        for i in range(y.shape[1]):
            annotation_channel = y[:, i]
            torchvision.utils.save_image(annotation_channel.unsqueeze(1), f"{folder}/annotation_channel_{i}_{idx}.png")


def prediction_to_image (pred, n_classes=8):
    
    shape = pred.shape
    image = torch.zeros(shape[0], shape[2], shape[3])
    if n_classes > 1:
        # Each channel is a class, each value grater than 0.5 is eaqual to the class id
        for i in range(n_classes):
            image[pred[:,i,:,:] == 1] = i
    return image
       

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers = 4, pin_memory = True):
    train_ds = CarvanaDataset (image_dir=train_dir, mask_dir=train_maskdir,n_classes=8, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers = num_workers, pin_memory=pin_memory, shuffle = True)
    val_ds = CarvanaDataset(image_dir= val_dir, mask_dir= val_maskdir,n_classes=8, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size = batch_size, num_workers = num_workers, pin_memory=pin_memory, shuffle = True )
    return train_loader, val_loader
def save_segmented_image(segmented_image, output_path):
    # Convertir l'image segmentée en un format approprié pour l'enregistrement
    segmented_image = segmented_image.squeeze().detach().cpu()
    segmented_image = TF.to_pil_image(segmented_image)

    # Enregistrer l'image segmentée
    segmented_image.save(output_path)
def get_image_paths(image_dir):
    image_paths = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".tif") :
            image_path = os.path.join(image_dir, filename)
            image_paths.append(image_path)
    return image_paths

