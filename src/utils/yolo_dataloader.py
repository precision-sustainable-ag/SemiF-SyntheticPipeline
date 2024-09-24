# this file creates a data loader to run yolov8

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image

image_dir = os.path.join(os.path.dirname(__file__),'../../projects/test_cutouts/TEST_small/results/images')
label_dir = os.path.join(os.path.dirname(__file__),'../../projects/test_cutouts/TEST_small/results/yolo_bbox_labels')

image_dir = os.path.abspath(image_dir)
label_dir = os.path.abspath(label_dir)

images = sorted([file for file in os.listdir(image_dir) if file.endswith('.jpg')])
labels = sorted([file for file in os.listdir(label_dir) if file.endswith('.txt')])

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

data = list(zip(images, labels))
random.shuffle(data)

train_size = int(train_ratio * len(data))
val_size = int(val_ratio * len(data))

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

train_images, train_labels = zip(*train_data)
val_images, val_labels = zip(*val_data)
test_images, test_labels = zip(*test_data)

class YOLOv8Dataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        if self.split == 'train':
            self.images = train_images
            self.labels = train_labels
        elif self.split == 'val':
            self.images = val_images
            self.labels = val_labels
        elif self.split == 'test':
            self.images = test_images
            self.labels = test_labels
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        transform = transforms.Compose([
            transforms.Resize((640,640)),
            transforms.ToTensor()
        ])

        img_name = self.images[idx]
        label_name = self.labels[idx]

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        
        # Load image using OpenCV
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        # Load label
        with open(label_path, 'r') as f:
            label = f.readlines()
        
        # Transform the label from YOLO format to PyTorch tensor (implement as needed)
        labels = self.process_label(label)
        
        images = []
        targets = []

        for img, target in zip(img,labels):
            images.append(img)  # Collect images
            targets.append(target)

        return img, targets

    def process_label(self, label):
        # Convert the YOLO .txt label into a tensor
        processed_labels = []
        for line in label:
            parts = list(map(float, line.strip().split()))
            processed_labels.append(parts)
        return torch.tensor(processed_labels)

# Create dataloaders for training, validation, and testing
def dataloader(batch_size=16):
    train_loader = DataLoader(YOLOv8Dataset(split='train'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(YOLOv8Dataset(split='val'), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(YOLOv8Dataset(split='test'), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
