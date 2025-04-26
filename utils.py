import pandas as pd
from pathlib import Path
import math
import random
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, random_split

class StreetViewDataset(Dataset):

    def __init__(self, labels_path, images_path, transform=None, device=None):
        super(StreetViewDataset, self).__init__()
        self.labels = pd.read_csv(labels_path)
        self.path = Path(images_path)
        self.image_list = sorted(self.path.glob("*.png"), key=lambda f: int(f.stem)) # since indicies are not continuous we read in the dir list
        self.transform = transform
        self.device = device

        assert len(self.image_list) == len(self.labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = self.image_list[index]
        image = decode_image(img_path).to(device=self.device)

        row = self.labels.iloc[index]
        label = torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32, device=self.device)

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
    train_DS = StreetViewDataset(train_labels, train_images)
    test_DS = StreetViewDataset(test_labels, test_images)

    train_size = int(config["split_tv"] * len(train_DS))
    val_size = len(train_DS) - train_size
    train_subset, val_subset = random_split(train_DS, [train_size, val_size])

    train = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test = DataLoader(test_DS, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    return train, val, test 



def haversine(preds, targets):
    lon1, lat1 = preds[:, 0], preds[:, 1]
    lon2, lat2 = targets[:, 0], targets[:, 1]

    # convert degrees to radians
    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    R = 6371  # Earth radius in km
    return R * c