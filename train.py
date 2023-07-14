from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
import torch.optim as optim
from segmentation_pytorch import UNet
from utils import (load_checkpoint, save_checkpoint, check_accuracy,get_loaders, save_predictions_as_imags)
#from semanticseg.DataDictModule import DataDictModule

## Hyperparameters 
LEARNIN_RATE =1E-6
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.999
GRADIENT_CLIPPING = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS= 200
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_IMG_DIR  = "E:\\images_multi_classes"
TRAIN_MASK_DIR  = "E:\\labels_multi_classes"
VAL_IMG_DIR  = "E:\\images_multi_val"
VAL_MASK_DIR  = "E:\\labels_multi_val"

def train_fn(loader, model, optimizer, grad_scaler, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop) : 
        data = data.to(device = DEVICE)
        targets = targets.float().to(device= DEVICE)
        #FORWARD 
        with torch.cuda.amp.autocast() : 
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #BACKWARD
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # update tqdm loop 
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
        A.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
        ],
    )

    n_classes=8
    model = UNet(n_channels=3, n_classes=n_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNIN_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)    
    train_loader, val_loader  = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR, 
        VAL_MASK_DIR, 
        BATCH_SIZE, 
        train_transform, 
        val_transform,
        NUM_WORKERS, 
        PIN_MEMORY,)


    

    #if LOAD_MODEL : 
        #load_checkpoint("E:\\checkpoint_5mm_352_images_multi_classes(8)_V1.pth.tar", model)
    check_accuracy(val_loader, model, device=DEVICE)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(NUM_EPOCHS) :
        train_fn(train_loader, model, optimizer, grad_scaler, loss_fn)
        
        #save the model 
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        print("##################",epoch,"##############################")
        #check_ accuracy 
        check_accuracy(val_loader,model, device=DEVICE)
        torch.save(model.state_dict(), "E:\\model_5mm_3576Im_multi_classes_new_scipt.pth.tar")

        #print some exemple to a folder
        save_predictions_as_imags(val_loader, model,folder ="E:\\resultat_predict_multi_classe\\", device= DEVICE )

if __name__ =="__main__" : 
    main()