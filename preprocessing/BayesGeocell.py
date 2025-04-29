# geocell_partition.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pickle

class GeoCell:
    def __init__(self, lat_min, lat_max, lon_min, lon_max, cell_id):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.cell_id = cell_id

    def contains(self, lat, lon):
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    def center(self):
        return (self.lat_min + self.lat_max)/2, (self.lon_min + self.lon_max)/2

class GeocellPartitioner:
    def __init__(self, threshold_mass=0.01):
        self.threshold_mass = threshold_mass
        self.cells = []

    def fit(self, lat_lon_train):
        # 1. Fit KDE
        self.kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        self.kde.fit(lat_lon_train)

        # 2. Build grid
        self.lat_grid = np.linspace(-90, 90, 360)
        self.lon_grid = np.linspace(-180, 180, 720)
        lonv, latv = np.meshgrid(self.lon_grid, self.lat_grid)
        self.latv = latv
        self.lonv = lonv
        self.grid_points = np.vstack([latv.ravel(), lonv.ravel()]).T

        log_density = self.kde.score_samples(self.grid_points)
        density = np.exp(log_density)
        self.density_grid = density.reshape(latv.shape)

        # 3. Partition
        self._split_region(-90, 90, -180, 180, 0)

    def _split_region(self, lat_min, lat_max, lon_min, lon_max, current_id):
        mask = (
            (self.latv >= lat_min) & (self.latv <= lat_max) &
            (self.lonv >= lon_min) & (self.lonv <= lon_max)
        )
        region_mass = self.density_grid[mask].sum()

        if region_mass < self.threshold_mass or len(self.cells) > 1000:
            cell = GeoCell(lat_min, lat_max, lon_min, lon_max, current_id)
            self.cells.append(cell)
            return

        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        split_lat = lat_range >= lon_range

        if split_lat:
            split = (lat_min + lat_max) / 2
            self._split_region(lat_min, split, lon_min, lon_max, current_id)
            self._split_region(split, lat_max, lon_min, lon_max, current_id+1)
        else:
            split = (lon_min + lon_max) / 2
            self._split_region(lat_min, lat_max, lon_min, split, current_id)
            self._split_region(lat_min, lat_max, split, lon_max, current_id+1)

    def assign_cell(self, lat, lon):
        for cell in self.cells:
            if cell.contains(lat, lon):
                return cell.cell_id
        return None

    def center_of_cell(self, cell_id):
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell.center()
        raise ValueError("Cell ID not found")
    
    def plot(self, save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geocell Partitions')

        for cell in self.cells:
            rect = plt.Rectangle(
                (cell.lon_min, cell.lat_min),
                cell.lon_max - cell.lon_min,
                cell.lat_max - cell.lat_min,
                linewidth=1,
                edgecolor='blue',
                facecolor='none'
            )
            ax.add_patch(rect)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved geocell plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Partitioner saved to {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            partitioner = pickle.load(f)
        print(f"Partitioner loaded from {filepath}")
        return partitioner