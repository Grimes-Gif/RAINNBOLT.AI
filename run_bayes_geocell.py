from preprocessing.BayesGeocell import GeocellPartitioner
from utils import plot_geocell_distribution
from pathlib import Path
import numpy as np
import pandas as pd

dir = Path("./Data/Geocells")

def save_lat_lon(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_lat_lon = train_df[['latitude', 'longitude']].to_numpy()
    test_lat_lon = test_df[['latitude', 'longitude']].to_numpy()

    dir.mkdir(parents=True, exist_ok=True)

    np.save(dir/'lat_lon_train.npy', train_lat_lon)
    np.save(dir/'lat_lon_test.npy', test_lat_lon)


save_lat_lon('./Data/labels/train_labels.csv', './Data/labels/test_labels.csv')
lat_lon_train = np.load(dir/'lat_lon_train.npy') 
# lat_lon_test = np.load(dir/'lat_lon_test.npy')
partitioner_train = GeocellPartitioner()
# partitioner_test = GeocellPartitioner()
partitioner_train.fit(lat_lon_train)
partitioner_train.plot(save_path="figures/bayesian_geocells_trainset.png")
print(len(partitioner_train.cells))
# partitioner_test.fit(lat_lon_test)
# partitioner_test.plot(save_path="figures/bayesian_geocells_testset.png")

plot_geocell_distribution(partitioner_train, lat_lon_train, save_path="./figures/bayesian_geocells_scatter")