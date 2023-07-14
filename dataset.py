import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as TF

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir,n_classes, transform=None) : 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.n_classes = n_classes
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self) : 
            return len(self.images)
        
    def __getitem__(self, index) :
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array (Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)
        label = np.zeros((self.n_classes, image.shape[0], image.shape[1]), dtype = np.float32)

        # Loop over the number of classes and match each pixel value with the label channel
        for i in range(self.n_classes) :
            label[i, mask == i] =1

        if self.transform is not None :
            augmentation = self.transform(image = image, mask = mask)
            image = augmentation["image"]
            mask = augmentation ["mask"]


        return image, label
class seg_data(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image = TF.to_tensor(image)
        return image
    
