import pandas as pd
from pathlib import Path
import math
import random
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split

class StreetViewDataLoader(Dataset):

    def __init__(self, labels_path, images_path, transform=None):
        super(StreetViewDataLoader, self).__init__()
        self.labels = pd.read_csv(labels_path)
        self.path = Path(images_path)
        self.image_list = sorted(self.path.glob("*.png"), key=lambda f: int(f.stem)) # since indicies are not continuous we read in the dir list
        self.transform = transform

        assert len(self.image_list) == len(self.labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = self.image_list[index]
        image = read_image(img_path)

        row = self.labels.iloc[index]
        label = torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label
    

def load_data(config, train_labels, test_labels, train_images, test_images):
    """
    Returns data loader for train and test class
    - config (dict):  
    - train_labels (str): path to the train labels
    - test_labels (str): path to the test labels
    - train_images (str): path to the training images directory
    - test_images (str): path to the testing images directory
    """
    train_DS = StreetViewDataLoader(train_labels, train_images)
    test_DS = StreetViewDataLoader(test_labels, test_images)

    train_size = int(config["split_tv"] * len(train_DS))
    val_size = len(train_DS) - train_size
    train_subset, val_subset = random_split(train_DS, [train_size, val_size])

    train = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test = DataLoader(test_DS, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    return train, val, test 
