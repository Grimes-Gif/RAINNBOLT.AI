import torch
import torch.nn as nn
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import load_data
from utils import haversine
from Models.MyCNN import BasicCNN
from preprocessing.BayesGeocell import GeocellPartitioner

class Trainer:
    def __init__(self, config, train_loader, val_loader, device=None, isGeocell=False):
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.lr = config["learning_rate"]
        self.epochs = config["epochs"]
        self.weight_decay = config["weight_decay"]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.haversine = haversine
        self.isGeocell = isGeocell

        if self.isGeocell:
            # Load raw lat-lon labels here
            lat_lon_train = np.load('lat_lon_train.npy')  # You should prepare this
            self.partitioner = GeocellPartitioner()
            self.partitioner.fit(lat_lon_train)

        self.train_losses = []
        self.val_losses = []


    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.isGeocell:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0

            for images, labels in tqdm(self.train_loader, desc=f"Evaluating train"):
                optimizer.zero_grad()
                if self.isGeocell:
                    # Map lat-lon labels to geocell class
                    lat, lon = labels[:, 0], labels[:, 1]
                    cell_ids = torch.tensor([self.partitioner.assign_cell(lat[i].item(), lon[i].item()) for i in range(len(lat))])
                    outputs = model(images)
                    loss = criterion(outputs, cell_ids.to(self.device))
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)

            val_loss = self.evaluate(loader=self.val_loader, mode='val', model=model)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, loader=None, mode='val', distance_threshold_km=25, model=None):
        """
        Evaluate model performance.
        
        Args:
            loader: DataLoader to evaluate on (default: self.val_loader)
            mode: 'val' during training, 'test' after training
            distance_threshold_km: threshold to compute precision
        """
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        within_threshold = 0
        distances = []

        if loader is None:
            loader = self.val_loader

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Evaluating ({mode})"):
                images, labels = images.to(self.device), labels.to(self.device)

                if self.isGeocell:
                    outputs = model(images)
                    pred_classes = outputs.argmax(dim=1)

                    # Get predicted (lat, lon) from geocell centers
                    pred_lat_lon = torch.tensor([self.partitioner.center_of_cell(cid.item()) for cid in pred_classes])
                    loss = criterion(outputs, torch.tensor([self.partitioner.assign_cell(lat.item(), lon.item()) for lat, lon in labels]))
                else:
                    outputs = model(images)

                    # Always compute MSE
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                if mode == 'test':
                    dist = self.haversine(outputs, labels)
                    distances.extend(dist.cpu().numpy())
                    within_threshold += (dist < distance_threshold_km).sum().item()
                    total_samples += images.size(0)

        avg_loss = total_loss / len(loader)

        if mode == 'val':
            self.val_losses.append(avg_loss)
            return avg_loss

        elif mode == 'test':
            distances = np.array(distances)
            accuracy = within_threshold / total_samples
            mean_error = distances.mean()
            median_error = np.median(distances)

            print(f"\n📊 Test Results:")
            print(f"  Mean Geodesic Error: {mean_error:.2f} km")
            print(f"  Median Geodesic Error: {median_error:.2f} km")
            print(f"  Accuracy @ {distance_threshold_km}km: {accuracy:.2%}")

            self.plot()
            return {
                'mean_error_km': mean_error,
                'median_error_km': median_error,
                'accuracy': accuracy
            }
        
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join('figures', 'loss_curve.png')
        plt.savefig(save_path)
        plt.close()

    def save_model(self, model, filename="model_weights.pth"):
        os.makedirs('./checkpoints', exist_ok=True)  # create folder if not exist
        save_path = os.path.join('checkpoints', filename)
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    with open("./environment_vars/master.yaml") as f:
        config = yaml.safe_load(f)
    train_labels = "./Data/labels/train_labels.csv"
    test_labels = "./Data/labels/test_labels.csv"
    train_images = "./Data/Streetview_Image_Dataset/train"
    test_images = "./Data/Streetview_Image_Dataset/test"
    net = BasicCNN() # place model here 
    train, val, test = load_data(config, train_labels, test_labels, train_images, test_images)
    trainer = Trainer(config, train, val)
    trainer.train(net)
    trainer.evaluate(loader=test, mode="test", model=net)