import os 
import os.path as path
import zipfile
import yaml
from huggingface_hub import snapshot_download
from huggingface_hub import list_repo_files
from huggingface_hub import hf_hub_download
import math

with open("./environment_vars/master.yaml") as f:
    vars = yaml.safe_load(f)
    local_dir = vars["local_dir"]
    percent_data = vars["percent_data"]
    
    f.close()

def list_data_stats(path):
    total_train_images = 0
    total_test_images = 0
    for (root,dir,files) in os.walk(path):
        if (("test" in root) and dir == []):
            total_test_images += len(os.listdir(root))
        elif (("train" in root) and dir == []):
            total_train_images += len(os.listdir(root))

    print("Total testing images: ", total_test_images)
    print("Total training images: ", total_train_images)

def unzip(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(os.path.join(root, file))

def get_data(dir, perct=100):
    # check if directory exists and is empty
    if (path.isdir(dir) == True):
        if (os.listdir(dir) != []):
            print("Clear directory before loading data", )
            return -1
    else:
        os.mkdir(dir)

    # get images
    if (perct == 100):
        print("Loading full data snapshot...")
        snapshot_download(repo_id="osv5m/osv5m", local_dir=dir, 
                        repo_type='dataset')
    else:
        # only download the percentage amount
        all_files = list_repo_files(repo_id="osv5m/osv5m", repo_type="dataset")
        zip_files = sorted([f for f in all_files if f.endswith(".zip")])

        # subtract 5 for testing set
        amt = math.floor((len(zip_files)-5) * (perct/100))
        selected_files = zip_files[:(amt + 5)]

        print(f"downloading {amt} files")

        for f in selected_files:
            hf_hub_download(
              repo_id="osv5m/osv5m",
              repo_type="dataset",
              filename=f,
              local_dir=dir
            )
            
    print("Data snapshot loaded, unzipping...")
    unzip(dir)
    print("finished! Data pushed to: ", dir)

def get_data_labels(dir):
    m_dir = dir + "/labels"
    os.mkdir(m_dir)
    print("Downloading data labels to: ", m_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type='dataset', local_dir=m_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir=m_dir)

get_data(local_dir, percent_data) # run dataloader
list_data_stats(local_dir + "/images")
get_data_labels(local_dir)