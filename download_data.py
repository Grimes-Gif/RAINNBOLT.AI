import os 
import yaml
import shutil
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def download():
    """
    Use API key to download dataset
    TODO: add try catch for api request and option to modify data if downloaded manually
    """
    with open("environment_vars/master.yaml") as f:
        vars = yaml.safe_load(f)
        local_dir = Path(vars["local_dir"])
        split = vars["split_tt"]
        f.close()

    api = KaggleApi()
    api.authenticate()

    local_dir.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files("ayuseless/streetview-image-dataset", path=local_dir, unzip=True)

    # move the labels to root directory
    labels_path = shutil.move(local_dir / "Streetview_Image_Dataset/coordinates.csv",
                               local_dir)
    """
    After download, prepare the data to be read by the data loader
    1.) Split into train and test data folders
    2.) Split csv into train and test labels
    """
    print("making test and train sets...")
    # create train and test data
    source_dir = local_dir / "Streetview_Image_Dataset"
    train_dir = source_dir / "train"
    test_dir = source_dir / "test"

    # create output dirs if needed
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = [f for f in source_dir.iterdir() if f.is_file()]
    
    split_idx = int(len(image_files) * split)
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    # Copy files to train
    for f in train_files:
        shutil.move(str(f), train_dir / f.name)

    # Copy files to test
    for f in test_files:
        shutil.move(str(f), test_dir / f.name)
    """
    Get labels for each dataset and write them to new csv
    """
    print("making test and train labels...")
    labels = pd.read_csv(labels_path)
    output_dir = Path(local_dir / "labels")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = os.listdir(local_dir / "Streetview_Image_Dataset/train")
    test_files = os.listdir(local_dir / "Streetview_Image_Dataset/test")

    train_labels = labels.iloc[sorted([int(ele.removesuffix(".png")) 
                             for ele in train_files])]
    test_labels = labels.iloc[sorted([int(ele.removesuffix(".png")) 
                            for ele in test_files])]
    
    pd.DataFrame(train_labels).to_csv(output_dir / "train_labels.csv", index=False)
    pd.DataFrame(test_labels).to_csv(output_dir / "test_labels.csv", index=False)

download()