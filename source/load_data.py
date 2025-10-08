import os
import torch
from torchvision.io import decode_image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

def fetch__data(pw):
    mnist = fetch_openml(name='mnist_784', version='active', data_home=pw, as_frame=False)
    return mnist

mnist = fetch__data('C:/Users/Abdou/Documents/Drive/repos/lenet_5/dataset')


data_path = './dataset' 
os.makedirs(data_path, exist_ok=True)

print(f"Type de l'objet global : {type(mnist)}")
print(f"Type des données (images) : {type(mnist.data)}")
print(f"Type des étiquettes (labels) : {type(mnist.target)}")

Image = torch.from_numpy(mnist.data.astype(np.float32))/ 255.0
label = torch.from_numpy(mnist.target.astype(np.int64))

print(f"Type des données (images) : {type(Image)}")
print(f"Type des étiquettes (labels) : {type(label)}")

print(Image[0,:])
plt.imshow(Image[0,:].reshape(28,28), cmap='gray') # Height, Width 
plt.show()



#Create DataLoader
class MNISTImageDataset(Dataset):
    def __init__(self, labels, imgs, transform=None, target_transform=None):
        self.img_labels = labels
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        image = self.imgs[idx,:].reshape(1,28,28)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
train_dataset = MNISTImageDataset(labels=label, imgs=Image)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#featch one batch 
#Print the shape of the images tensor and the labels tensor
images_batch, labels_batch = next(iter(train_dataloader))
    
# Print the shape of the images tensor and the labels tensor
print("\n--- Batch Information ---")
print(f"Shape of the images tensor: {images_batch.shape}")
print(f"Shape of the labels tensor: {labels_batch.shape}")