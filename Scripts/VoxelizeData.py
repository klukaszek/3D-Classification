import os
import torch
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trimesh.voxel.creation import voxelize_binvox
from trimesh.exchange.binvox import load_binvox
from torch.utils.data import TensorDataset, DataLoader

def normalize_0_1(data):
    """
    Normalize data in an array to be between 0 and 1 (in our case, 0 or 1)
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class Voxelize():
    """
    Class for voxelizing .off files, ply files, etc. and storing them as numpy arrays
    """
    def __init__(self, path, dataset, system='Unix'):
        allowed_datasets = ['ModelNet40', 'ShapeNetCore.v2', 'toys4k_point_clouds']

        allowed_OS = ['Unix', 'Windows']

        if dataset not in allowed_datasets:
            raise ValueError(f'Dataset not supported. Please choose from: {allowed_datasets}')
        
        if system not in allowed_OS:
            raise ValueError(f'OS not supported. Please choose from: {allowed_OS}')

        
        self.path = path
        self.dataset = dataset
        self.system = system
        self.files = self.get_files()
        self.train_labels = self.files[self.files['split'] == 'train']['class'].tolist()
        self.test_labels = self.files[self.files['split'] == 'test']['class'].tolist()
        self.train = self.voxelize('train')


    def get_files(self):
        """
        Returns a dataframe containing the paths to the .off files and their corresponding labels
        """
        if self.dataset == "ModelNet40":
            metadata = pd.read_csv(self.path + 'metadata_modelnet40.csv')
            metadata.drop(columns=['object_id'], inplace=True)
            return metadata

    def voxelize(self, split):
        """
        Voxelize the meshes in the dataset and return a list of voxel grids
        """
        allowed_splits = ['train', 'test']

        if split not in allowed_splits:
            raise ValueError('Split not supported. Please choose from: {}'.format(allowed_splits))
        
        if self.dataset == "ModelNet40":
            # 
            split_files = self.files[self.files['split'] == split]

            voxelized_meshes = []
            
            labels = split_files['class'].unique().tolist()

            n = 0

            for label in labels:
                files = split_files[split_files['class'] == label]
                files = files['object_path'].tolist()

                print(files)

                i = 0

                print(f'Voxelizing {split} meshes of the {label} class...')

                # Iterate through each .off file
                for file in files:
                    
                    # Get path to .off file
                    path = self.path + 'ModelNet40/' + file

                    args = ['./cuda_voxelizer', '-f', path, '-s', '32', '-o', 'binvox']
                    
                    # Run the cuda_voxelizer executable to create a binvox file (This is orders of magnitude faster than using trimesh4.0)
                    popen = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    popen.wait()

                    if popen.returncode != 0:
                        raise subprocess.CalledProcessError(popen.returncode, popen.args)
                    
                    if popen.stderr:
                        print(popen.stderr)

                    # Get path to binvox file that was created from the .off file
                    binvox_path = path + '_32.binvox'

                    # Open the binvox file
                    binvox = open(binvox_path, 'rb')

                    # Load the binvox file
                    voxel_grid = load_binvox(binvox)

                    # Close the binvox file
                    binvox.close()

                    # Delete the binvox file
                    os.remove(binvox_path)

                    # Extract the voxel grid as a numpy array
                    voxel_grid = voxel_grid.matrix.astype(np.int8)

                    # Normalize the voxel grid to be either 0 and 1
                    voxel_grid = normalize_0_1(voxel_grid)

                    # Add voxel grid to list of voxel grids
                    voxelized_meshes.append(voxel_grid)

                    i+=1

                n += 1
                print(f'Finished voxelizing {i} {split} meshes in the {label} class. ({n}/{len(labels)})')

            return voxelized_meshes
        
    def build_dataloader(self, split):
        """
        Build a dataloader for the voxelized meshes
        """
        allowed_splits = ['train', 'test']

        if split not in allowed_splits:
            raise ValueError('Split not supported. Please choose from: {}'.format(allowed_splits))

        # Check if GPU is available
        gpu_memory = False
        if torch.cuda.is_available():
            gpu_memory = True
        
        if split == 'train':
            data = self.train
            labels = self.train_labels

            train_dataset = TensorDataset(data, labels)

            # VoxNet paper uses batch size of 32
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=gpu_memory)
            return train_loader

        elif split == 'test':
            data = self.test
            labels = self.test_labels

            test_dataset = TensorDataset(data, labels)

            # VoxNet paper uses batch size of 32
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=gpu_memory)
            return test_loader
    
    def render(self, data):
        """
        Render a voxel grid as a 3D plot
        """
        # make plot
        fig = plt.figure()
        
        # make 3D axis
        ax = fig.add_subplot(111, projection='3d')

        # plot voxels
        ax.voxels(data, edgecolor="k")

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        plt.show()

        plt.clf()


