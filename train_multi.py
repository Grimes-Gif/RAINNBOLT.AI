import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils_multi import load_data
from utils_multi import haversine, generate_gradcam
# from Models.basic_cnn import BasicCNN 
from Models.resnet_18 import BackboneResNet
from Models.basic_vit import BasicClipEncoder
from Models.pigeon_multi import PigeonNet
from Models.ViT import ViTGeoLocalization
from transformers import CLIPImageProcessor
import torchvision.transforms as T
import os

class Trainer:
    def __init__(self, config, train_loader, val_loader, device=None):
        self.batch_size = config["batch_size"]
        self.num_workers = config['num_workers']
        self.lr = config['alpha']
        self.epochs = config['epochs']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_geocelled = config['is_geocelled']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.haversine = haversine
        self.c_beta = config['c_beta']
        self.g_beta = config['g_beta']
        self.r_beta = config['r_beta']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']

        self.train_losses = []
        self.reg_val_losses = []
        self.g_train_losses = []
        self.g_val_losses = []
        self.c_train_losses = []
        self.c_val_losses = []
        self.gradient_norms = []
        
        self.num_coarse_geocells = config['coarse_geocells']
        self.num_geocells = config['particular_geocells']

    def train(self, model):
        model.to(self.device)
        print(self.device)
        
        # scaler = GradScaler(device='cuda')
        
        class_criterion = torch.nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': self.lr * 0.1},
            {'params': model.neck.parameters(), 'lr': self.lr},
            {'params': model.coarse_geocell_head.parameters(), 'lr': self.lr},
            {'params': model.geocell_head.parameters(), 'lr': self.lr},
            {'params': model.offset_head.parameters(), 'lr': self.lr * .1},
        ], lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            threshold=1e-4
        )

        for epoch in range(self.epochs):
            progress = epoch / self.epochs

            c_beta = max(0.1, .5 - progress)
            g_beta = max(0.3, .4 - progress * .5)
            r_beta = .5
            
            model.train()
            running_loss = 0.0

            for images, coarse_geocell_labels, geocell_labels, coords in tqdm(self.train_loader, desc=f"Evaluating train"):
                images = images.to(self.device)
                coarse_geocell_labels = coarse_geocell_labels.squeeze().to(self.device)
                geocell_labels = geocell_labels.squeeze().to(self.device)
                coords = coords.to(self.device)
                
                optimizer.zero_grad()
                
                coarse_geocell_logits, geocell_logit, dcoords = model(images)

                coarse_loss = class_criterion(coarse_geocell_logits, coarse_geocell_labels)
                class_loss = class_criterion(geocell_logit, geocell_labels)
                reg_loss = regression_criterion(dcoords, coords)
                
                loss = c_beta * coarse_loss + g_beta * class_loss + r_beta * reg_loss 
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)

            val_loss = self.evaluate(loader=self.val_loader, mode='val', model=model)
            scheduler.step(val_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            self.log_gradients(model, step_name=f"Epoch {epoch+1}/{self.epochs} (post-update)")

    def evaluate(self, loader=None, mode='val', distance_threshold_km=200, model=None):
        """
        Evaluate model performance.
        
        Args:
            loader: DataLoader to evaluate on (default: self.val_loader)
            mode: 'val' during training, 'test' after training
            distance_threshold_km: threshold to compute precision
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        within_threshold = 0
        within_20_threshold = 0
        within_100_threshold = 0
        
        c_total_loss = 0.0
        c_within_threshold = 0
        
        g_total_loss = 0.0
        g_within_threshold = 0
        distances = []

        if loader is None:
            loader = self.val_loader

        with torch.no_grad():
            for images, c_geo_labels, geocell_labels, true_coords in tqdm(loader, desc=f"Evaluating ({mode})"):
                
                images = images.to(self.device)
                geocell_labels = geocell_labels.to(self.device)
                c_geo_labels = c_geo_labels.to(self.device)
                true_coords = true_coords.to(self.device)
                
                coarse_geocell_logits, geocell_logits, delta_coords = model(images)

                # Always compute MSE if it is not geocelled
                outputs = delta_coords
                labels = true_coords
                criterion = nn.MSELoss()
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                dist = self.haversine(outputs, labels)
                distances.extend(dist.cpu().numpy())
                within_threshold += (dist < distance_threshold_km).sum().item()
                within_20_threshold += (dist < 20).sum().item()
                within_100_threshold += (dist < 100).sum().item()
                total_samples += images.size(0)
                
                # geo celled criterion
                g_labels = geocell_labels.squeeze()
                g_criterion = nn.CrossEntropyLoss()
                g_loss = g_criterion(geocell_logits, g_labels)
                g_total_loss += g_loss.item()
                g_prediction = geocell_logits.argmax(dim=1)
                g_correct = (g_prediction == g_labels).int()
                g_within_threshold += g_correct.sum().item()
                
                # coars celled criterion
                c_labels = c_geo_labels.squeeze()
                c_criterion = nn.CrossEntropyLoss()
                c_loss = c_criterion(coarse_geocell_logits, c_labels)
                c_total_loss += c_loss.item()
                c_prediction = coarse_geocell_logits.argmax(dim=1)
                c_correct = (c_prediction == c_labels).int()
                c_within_threshold += c_correct.sum().item()

        avg_loss = total_loss / len(loader)
        g_avg_loss = g_total_loss / len(loader)
        c_avg_loss =  c_total_loss / len(loader)

        if mode == 'val':
            self.reg_val_losses.append(avg_loss)
            self.g_val_losses.append(g_avg_loss)
            self.c_val_losses.append(c_avg_loss)
            return avg_loss

        elif mode == 'test':
            distances = np.array(distances)
            accuracy = (within_threshold * 1.0/ total_samples)
            accuracy_20 = (within_20_threshold * 1.0/ total_samples)
            accuracy_20 = (within_100_threshold * 1.0/ total_samples)
            g_accuracy = (g_within_threshold * 1.0/ total_samples)
            c_accuracy = (c_within_threshold * 1.0/ total_samples)
            mean_error = distances.mean()
            median_error = np.median(distances)

            print(f"\n📊 Test Results:")
            print(f"  Mean @ 20km Geodesic Error: {mean_error:.2f} km")
            print(f"  Median @ 20km Geodesic Error: {median_error:.2f} km")
            print(f"  Accuracy @ 20 km: {accuracy_20:.2%}")
            print(f"  Accuracy @ {distance_threshold_km}km: {accuracy:.2%}")
            print(f"  Accuracy with {self.num_geocells} particular cells: {g_accuracy:.2%}")
            print(f"  Accuracy with {self.num_coarse_geocells} coarse cells: {c_accuracy:.2%}")

            self.plot()
            self.plot_gradient_norms()
            
            return {
                'mean_error_km': mean_error,
                'median_error_km': median_error,
                'accuracy': accuracy
            }
            
        
    def plot(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(self.train_losses, label='Train Loss', color='tab:blue')
        ax1.plot(self.reg_val_losses, label='Regression Validation Loss', color='tab:cyan')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE/Combined Losses', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(self.c_val_losses, label='Coarse Geocell Val Loss', color='#FFA500')
        ax2.plot(self.g_val_losses, label='Geocell Val Loss', color='#FF9730')
        ax2.set_ylabel('Cross-Entropy Losses', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()

        # Make sure both legends show
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        os.makedirs('figures', exist_ok=True)
        save_path = os.path.join('figures', 'loss_curve.png')
        plt.style.use("seaborn-v0_8")
        plt.savefig(save_path)
        plt.close()

    def save_model(self, model, filename="model_weights.pth"):
        os.makedirs('./checkpoints', exist_ok=True)
        save_path = os.path.join('checkpoints', filename)
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
        
    def grad_cams(self, model, train_loc=''):
        base_path = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/Data/Streetview_Image_Dataset/test/'
        images_considered = ['7.png', '9.png', '13.png', '23839.png', '3166.png', '3209.png', '3817.png', '3877.png']
        for i, image in enumerate(images_considered): 
            for j in range (2, 4): 
                image_path = base_path + image
                target_layer = model.backbone.blocks[-1]
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir="g_figures" + train_loc,
                    device="cuda",
                    head='geo'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir="r_figures" + train_loc,
                    device="cuda",
                    head='coarse'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir="c_figures" + train_loc,
                    device="cuda",
                    head='coords'
                )
                target_layer = model.backbone.blocks[-1].norm
                path1 = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/figures/' + "g_figures_low" + str(j)
                path2 = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/figures/' + "figures_low" + str(j)
                path3 = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/figures/' + "c_figures_low" + str(j)
                # os.makedirs(path1)
                # os.makedirs(path2)
                # os.makedirs(path3)
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir=path1,
                    device="cuda",
                    head='geo'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir=path2,
                    device="cuda",
                    head='coords'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir=path3,
                    device="cuda",
                    head='coords'
                )
    def grad_cams(self, model, train_loc=''):
        base_path = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/Data/Streetview_Image_Dataset/test/'
        images_considered = ['7.png', '9.png', '13.png', '23839.png', '3166.png', '3209.png', '3817.png', '3877.png']
        for i, image in enumerate(images_considered): 
            for j in range (2, 4): 
                image_path = base_path + image
                target_layer = model.backbone.features[-2]
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir="g_figures",
                    device="cuda",
                    head='geo'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir="r_figures",
                    device="cuda",
                    head='coarse'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir="c_figures",
                    device="cuda",
                    head='coords'
                )
                target_layer = model.backbone.features[j]
                path1 = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/figures/' + "g_figures_low" + str(j)
                path2 = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/figures/' + "figures_low" + str(j)
                path3 = '/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/scratch/RAINNBOLT.AI/figures/' + "c_figures_low" + str(j)
                # os.makedirs(path1)
                # os.makedirs(path2)
                # os.makedirs(path3)
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir=path1,
                    device="cuda",
                    head='geo'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir=path2,
                    device="cuda",
                    head='coords'
                )
                generate_gradcam(
                    model=model,
                    image_path=image_path,
                    target_layer=target_layer,
                    save_dir=path3,
                    device="cuda",
                    head='coords'
                )
        
    # 7 has a asphalt road, 9 has a dirt road, 93 has some shacks
    def log_gradients(self, model, step_name=""):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        print(f"{step_name} | Gradient norm: {total_norm:.4f}")
        
    def plot_gradient_norms(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.gradient_norms, label="Gradient Norm")
        plt.xlabel("Training Step")
        plt.ylabel("L2 Norm of Gradients")
        plt.title("Gradient Norms Over Training")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/gradient_norms.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_labels="/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/Data/labels/train_labels_300_3988.csv"
    test_labels="/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/Data/labels/test_labels_300_3988.csv"
    train_images="/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/Data/Streetview_Image_Dataset/train"
    test_images="/home/hice1/dvarzari3/scratch/RAINNBOLT.AI/Data/Streetview_Image_Dataset/test"
    
    backbone = BackboneResNet()
    # backbone = BasicClipEncoder()
    
    net = PigeonNet(backbone=backbone, embedding_dim=512, num_coarse_geocells=300, num_geocells=3988, dropout=config['dropout']) # place model here 
    # net = ViTGeoLocalization(num_geocells=3988, output_dim=2, dropout=.2)
    print('created model')
    
    print('transfrom')
    val = 224
    print(val)
    #
    vit_transform = T.Compose([
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train, val, test = load_data(config, train_labels, test_labels, train_images, test_images, transform=vit_transform)
    trainer = Trainer(config, train, val)
    trainer.train(net)
    trainer.evaluate(test, mode="test", model=net)
    trainer.save_model(net, filename="res_test.pth")
    trainer.grad_cams(net)
    

    
    
