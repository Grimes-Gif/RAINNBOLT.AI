import os
import pandas as pd
import torch
import yaml
from utils import StreetViewDataLoader
from utils import load_data

class Trainer:
    def __init__(self, config, device=None):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.lr = config.alpha
        self.epochs = config.epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = config.dataset

    def train(self, model, dataloader):
        model.to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        for epoch in range(self.epochs):
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
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    with open("./environment_vars/master.yaml") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config, torch.get_device())
    dl = trainer.load_data()
    # trainer.train(model, dl)