import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
        # decode_image reads in 8 bit integers so we need to convert to float32
        image = decode_image(img_path).to(device=self.device).to(torch.float32)/255 

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

def generate_gradcam(model, image_path, target_layer, save_dir='figures', device='cuda', output_index=0):
    """
    Generates GradCAM visualization for a given image and model.

    Args:
        model (torch.nn.Module): The model to analyze.
        image_path (str or Path): Path to input image.
        target_layer (torch.nn.Module): The layer you want to visualize (must be a Conv2d).
        save_dir (str): Folder to save the figure.
        device (str): 'cuda' or 'cpu'.
        output_index (int): For multi-output models, which output to visualize (default 0).
    """

    # Setup
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor()
    ])
    image = decode_image(str(image_path)).to(device)  # [C,H,W] uint8
    image = image.float() / 255.0  # Normalize to [0,1]
    input_tensor = transform(image).unsqueeze(0)  # [1,C,H,W]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=='cuda'))

    # Generate GradCAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]  # remove batch dimension

    # Convert input image for visualization
    input_numpy = input_tensor.squeeze(0).permute(1,2,0).cpu().numpy()  # [H,W,C]
    input_numpy = (input_numpy - input_numpy.min()) / (input_numpy.max() - input_numpy.min())

    # Overlay GradCAM on input
    visualization = show_cam_on_image(input_numpy, grayscale_cam, use_rgb=True)

    # Plot and save
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"gradcam_{filename}")

    plt.figure(figsize=(8,8))
    plt.imshow(visualization)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"GradCAM saved to {save_path}")

def plot_geocell_distribution(partitioner, labels, save_path):
    """
    partitioner: your GeocellPartitioner object
    labels: numpy array of shape (N, 2) with (lat, lon) for each image
    """

    # Step 1: Assign each image to a geocell ID
    cell_ids = np.array([
        partitioner.assign_cell(lat, lon) for lat, lon in labels
    ])

    # Step 2: Count number of images per geocell
    unique_ids, counts = np.unique(cell_ids, return_counts=True)

    # Step 3: Get center of each geocell
    centers = np.array([
        partitioner.center_of_cell(cell_id) for cell_id in unique_ids
    ])

    latitudes = centers[:, 0]
    longitudes = centers[:, 1]

    # Step 4: Scatter plot
    plt.figure(figsize=(14, 7))
    sc = plt.scatter(
        longitudes, latitudes,
        c=counts,         # Color by count
        s=counts,         # Size by count (may scale it if needed)
        cmap='viridis',   # Color map
        alpha=0.75,
        edgecolors='k'
    )
    plt.colorbar(sc, label="Image Count")
    plt.title("Geocell Distribution by Image Count")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
