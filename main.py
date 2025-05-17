import torch
import shutil
from torch import nn
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from PIL import Image
import PIL.Image
from IPython.display import display
from torchvision import models

epochs = 30
learning_rate = 0.01
batch_size = 128
num_workers = os.cpu_count()
hidden_units = 10
loss_function = nn.CrossEntropyLoss()
drop_rate = 0.5
num_photos = 1000
num_augmentations = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

!gdown 'https://drive.google.com/uc?export=download&id=1qcc8hUU0FCCJ2dsgkBMHVHXUF_uMmlk2' -O /content/3Food.tar.gz && tar -xzvf /content/3Food.tar.gz -C /content/ > /dev/null 2>&1


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

shit_pizza_photos = [
    "3Food/pizza/1047561.jpg", "3Food/pizza/1069629.jpg", "3Food/pizza/1075568.jpg",
    "3Food/pizza/1234172.jpg", "3Food/pizza/1407753.jpg", "3Food/pizza/1454995.jpg",
    "3Food/pizza/1512514.jpg", "3Food/pizza/1652943.jpg", "3Food/pizza/166823.jpg", "3Food/pizza/1671531.jpg",
    "3Food/pizza/1980167.jpg", "3Food/pizza/2265512.jpg", "3Food/pizza/2451169.jpg", "3Food/pizza/2476468.jpg",
    "3Food/pizza/2486277.jpg", "3Food/pizza/2576168.jpg", "3Food/pizza/2687575.jpg",
    "3Food/pizza/2693334.jpg", "3Food/pizza/2739039.jpg", "3Food/pizza/2754150.jpg", "3Food/pizza/2774899.jpg",
    "3Food/pizza/2785084.jpg", "3Food/pizza/3055697.jpg", "3Food/pizza/32666.jpg", "3Food/pizza/3398309.jpg",
    "3Food/pizza/3401767.jpg", "3Food/pizza/3803596.jpg", "3Food/pizza/3821701.jpg", "3Food/pizza/3826377.jpg",
    "3Food/pizza/625687.jpg", "3Food/pizza/626170.jpg", "3Food/pizza/790432.jpg"
]

shit_steak_photos = [
    "3Food/steak/1093966.jpg", "3Food/steak/1117936.jpg", "3Food/steak/1212161.jpg",
    "3Food/steak/1324791.jpg", "3Food/steak/1340977.jpg", "3Food/steak/179293.jpg", "3Food/steak/1822407.jpg",
    "3Food/steak/217250.jpg", "3Food/steak/2788759.jpg", "3Food/steak/2890573.jpg", "3Food/steak/3204977.jpg",
    "3Food/steak/3425047.jpg", "3Food/steak/3895825.jpg"
]

shit_sushi_photos = [
    "3Food/sushi/1022922.jpg", "3Food/sushi/1033302.jpg", "3Food/sushi/1073711.jpg",
    "3Food/sushi/1446129.jpg", "3Food/sushi/1484631.jpg", "3Food/sushi/3228675.jpg", "3Food/sushi/3499178.jpg",
    "3Food/sushi/4592.jpg", "3Food/sushi/492302.jpg", "3Food/sushi/769781.jpg"
]

for path in shit_pizza_photos:
    path = Path(path)
    try:
        path.unlink()
    except:
        continue

for path in shit_steak_photos:
    path = Path(path)
    try:
        path.unlink()
    except:
        continue

for path in shit_sushi_photos:
    path = Path(path)
    try:
        path.unlink()
    except:
        continue

subset_data = datasets.ImageFolder("/content/3Food")

train_size = int(len(subset_data) * 0.8)
test_size = len(subset_data) - train_size

train_subset_from_split, test_subset_from_split = torch.utils.data.random_split(dataset= subset_data, lengths=[train_size, test_size])

num_augmentations = 5

class AugmentedMultipliedDataset(Dataset):
    def __init__(self, original_subset, train_transform, test_transform, num_augmentations):
        self.original_subset = original_subset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_augmentations = num_augmentations
        self.num_versions_per_image = num_augmentations + 1

    def __len__(self):
        return len(self.original_subset) * self.num_versions_per_image

    def __getitem__(self, index):
        original_image_idx = index // self.num_versions_per_image
        version_idx = index % self.num_versions_per_image

        img, label = self.original_subset[original_image_idx]

        if version_idx < self.num_augmentations:
            transformed_img = self.train_transform(img)
        else:
            transformed_img = self.test_transform(img)

        return transformed_img, label

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_dataset_final = AugmentedMultipliedDataset(
    original_subset=train_subset_from_split,
    train_transform=train_transform,
    test_transform=test_transform,
    num_augmentations=num_augmentations
)

test_dataset_final = TransformedSubset(
    subset=test_subset_from_split,
    transform=test_transform
)

train_loader = DataLoader(train_dataset_final,
                          batch_size= batch_size,
                          shuffle= True,
                          num_workers= num_workers)
test_loader = DataLoader(test_dataset_final,
                         batch_size= batch_size,
                         shuffle= False,
                         num_workers= num_workers)

class FoodClassifier(nn.Module):
    def __init__(self, in_channels = 3, classes = 3, hidden_units= hidden_units, drop_rate = drop_rate):
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop_rate))
        self.convblock2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop_rate))
        self.Linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(hidden_units*16*16, classes))
        self.Model = nn.Sequential(self.convblock1,
                                   self.convblock2,
                                   self.Linear)
    def forward(self, x):
        return self.Model(x)

model = FoodClassifier()

def trainingandtesting(Model,
                       train_loader,
                       test_loader,
                       epochs = 5,
                       learning_rate = 0.01,
                       device = device,
                       optimizer = None,
                       loss_function = None,
                       lr_scheduler = None,
                       Model_state_dict = None):

    import time
    from tqdm import tqdm
    Model.to(device)
    if optimizer == None:
        optimizer = torch.optim.AdamW(params= Model.parameters(), lr= learning_rate, weight_decay= 0.01)

    if loss_function == None:
        loss_function = nn.CrossEntropyLoss()

    if lr_scheduler == None:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                    max_lr= learning_rate,
                                    steps_per_epoch= len(train_loader),
                                    epochs= epochs )

    if not Model_state_dict == None:
        Model.load_state_dict(torch.load(Model_state_dict))

    num_image = len(test_loader.dataset)

    train_losses_per_epoch = []
    test_losses_per_epoch = []
    right_answers = 0
    start_time = time.time()

    for epoch in tqdm(range(epochs), desc="Training", ncols=100):
        train_total_loss = 0
        test_total_loss = 0
        right_answers = 0

        Model.train()
        for X,y in train_loader:
            X = X.to(device)
            y = y.to(device)
            pred = Model(X)
            loss = loss_function(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_total_loss += loss.item()

        with torch.inference_mode():
            Model.eval()
            for X,y in test_loader:
                X.to(device)
                y.to(device)
                pred = Model(X)
                loss = loss_function(pred, y)
                test_total_loss += loss.item()
                if epoch == epochs - 1:
                    right_answers += torch.sum(torch.eq(pred.argmax(dim = 1), y)).item()

        train_loss_per_epoch = train_total_loss/len(train_loader)
        train_losses_per_epoch.append(train_loss_per_epoch)

        test_loss_per_epoch = test_total_loss/len(test_loader)
        test_losses_per_epoch.append(test_loss_per_epoch)

    torch.save(Model.state_dict(), "Model_last.pth")

    plt.plot(range(epochs), train_losses_per_epoch, c= "r", label= "train loss curve per epoch")
    plt.plot(range(epochs), test_losses_per_epoch, c= "b", label= "test loss curve per epoch")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    accuracy = right_answers/num_image*100
    print(f"Accuracy: {accuracy:.2f}%")
    print("Your model state dict has been saven in 'Model_last.pth'")
trainingandtesting(model, train_loader, test_loader,epochs = 1)

len(train_loader)
