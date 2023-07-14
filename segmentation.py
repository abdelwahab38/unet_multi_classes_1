import torch
import os
from dataset import seg_data
from torch.utils.data import DataLoader
from utils import  get_image_paths
import torch.nn as nn
from segmentation_pytorch import UNET
from PIL import Image
from torchvision import transforms
from osgeo import gdal
from utils import prediction_to_image

model = UNET(in_channels=3, out_channels=8)

image_dir ="E:\\ortho-3419\\images"
model_path = "E:\\checkpoint_5mm_352Im.pth.tar"
output_dir = "E:\\ortho-3419\\sortie_seg"

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

image_paths = get_image_paths(image_dir)
test_dataset = seg_data(image_paths)
print (image_paths)


image_filenames = os.listdir(image_dir)
for image_filename in image_filenames:
    # Chemin vers l'image d'entrée
    image_path = os.path.join(image_dir, image_filename)

    # Charger l'image et effectuer les transformations nécessaires
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Effectuer la segmentation sur l'image avec le modèle
    with torch.no_grad():
        segmented_image = model(image)


    # Convertir les valeurs de segmentation en une image binaire
    threshold = 0.5
    segmented_image = (segmented_image > threshold).float()
    segmented_image=prediction_to_image(segmented_image,n_classes=8)

    # Enregistrer l'image segmentée en tant qu'image temporaire
    temp_image_path = "temp_image.tif"
    transforms.ToPILImage()(segmented_image.squeeze()).save(temp_image_path)

    # Ouvrir l'image source avec GDAL pour extraire les informations de coordonnées
    src_dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    src_geotransform = src_dataset.GetGeoTransform()
    src_projection = src_dataset.GetProjection()

    # Ouvrir l'image de destination avec GDAL pour écrire les informations de coordonnées
    dst_dataset = gdal.Open(temp_image_path, gdal.GA_Update)
    dst_dataset.SetGeoTransform(src_geotransform)
    dst_dataset.SetProjection(src_projection)

    src_dataset = None
    dst_dataset.FlushCache()
    dst_dataset = None

    # Renommer l'image temporaire en tant que fichier final
    output_path = os.path.join(output_dir, image_filename)
    os.rename(temp_image_path, output_path)

