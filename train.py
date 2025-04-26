import torch
import torch.nn as nn
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from utils import haversine

class Trainer:
    def __init__(self, network, config, train_loader, val_loader, device=None):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.lr = config.alpha
        self.epochs = config.epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.net = network
        self.haversine = haversine

        self.train_losses = []
        self.val_losses = []

    def train(self, model):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        for epoch in range(self.epochs):
            self.net.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)

            val_loss = self.evaluate(loader=self.val_loader, mode='val')
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, loader=None, mode='val', distance_threshold_km=25):
        """
        Evaluate model performance.
        
        Args:
            loader: DataLoader to evaluate on (default: self.val_loader)
            mode: 'val' during training, 'test' after training
            distance_threshold_km: threshold to compute precision
        """
        self.net.eval()
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
                outputs = self.model(images)

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
        plt.show()


if __name__ == "__main__":
    with open("./environment_vars/master.yaml") as f:
        config = yaml.safe_load(f)
    train_labels = "./Data/labels/train_labels.csv"
    test_labels = "./Data/labels/test_labels.csv"
    train_images = "./Data/Streetview_Image_Dataset/train"
    test_images = "./Data/Streetview_Image_Dataset/test"
    train, val, test = load_data(config, train_labels, test_labels, train_images, test_images)
    trainer = Trainer(config, train)
    # trainer.train(model, dl)