import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OSV5MDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, limit=None):
        self.data = pd.read_csv(csv_path)
        if limit:
            self.data = self.data.iloc[:limit]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['file_path'])  # assuming there's a 'file_path' column
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Assuming lat/lon labels are in the columns 'latitude', 'longitude'
        label = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
        return image, label

class Trainer:
    def __init__(self, config, device=None):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self, image_root, csv_path, limit=250000):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = OSV5MDataset(csv_path, image_root, transform=transform, limit=limit)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def train(self, model, dataloader, epochs=10, lr=1e-4):
        model.to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")