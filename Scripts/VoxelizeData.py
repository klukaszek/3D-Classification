import os
import torch
import subprocess
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trimesh.voxel.creation import voxelize_binvox
from trimesh.exchange.binvox import load_binvox
from torch.utils.data import TensorDataset, DataLoader

"""
Author: Kyle Lukaszek

File for voxelizing .off files, ply files, etc. and storing them as numpy arrays
Includes a function for rendering a voxel grid as a 3D plot, and saving voxel grids as .npz files
"""

class Voxelize():
    """
    Class for voxelizing .off files, ply files, etc. and storing them as numpy arrays
    """
    def __init__(self, path, dataset, system='Unix', render=False, save=False, save_path=None, overwrite=False):
        allowed_datasets = ['ModelNet40', 'ShapeNetCore.v2', 'Toys4K']

        allowed_OS = ['Unix', 'Windows']

        if dataset not in allowed_datasets:
            raise ValueError(f'Dataset not supported. Please choose from: {allowed_datasets}')
        
        if system not in allowed_OS:
            raise ValueError(f'OS not supported. Please choose from: {allowed_OS}')

        print(f'Voxelizing {dataset} dataset...')
        
        self.path = path
        self.dataset = dataset
        self.system = system

        # Get a dataframe containing the paths to the .off files and their corresponding labels
        self.files = self.get_files()

        # Get the labels for the train and test sets
        self.train_labels = self.files[self.files['split'] == 'train']['class'].tolist()
        self.test_labels = self.files[self.files['split'] == 'test']['class'].tolist()

        # Voxelize the meshes in the dataset
        self.train = self.voxelize('train', render)
        if save:
            save_voxels(self.train, self.train_labels, save_path=save_path + 'Train.npz', overwrite=overwrite)
        
        self.test = self.voxelize('test', render)
        if save:
            save_voxels(self.test, self.test_labels, save_path=save_path + 'Test.npz', overwrite=overwrite)


    def get_files(self):
        """
        Returns a dataframe containing the paths to the .off files and their corresponding labels
        """
        if self.dataset == "ModelNet40":
            metadata = pd.read_csv(self.path + 'metadata_modelnet40.csv')
            metadata.drop(columns=['object_id'], inplace=True)
            return metadata
        if self.dataset == "Toys4K":
            metadata = pd.read_csv(self.path + 'metadata_toys4k.csv')
            metadata.drop(columns=['object_id'], inplace=True)

            # This mesh is corrupted and is also huge so we will drop it
            metadata.drop(metadata[metadata['object_path'] == 'octopus/octopus_004/mesh.obj'].index, inplace=True)
            return metadata

    def voxelize(self, split, render=False):
        """
        Voxelize the meshes in the dataset and return a list of voxel grids
        """
        allowed_splits = ['train', 'test']

        if split not in allowed_splits:
            raise ValueError('Split not supported. Please choose from: {}'.format(allowed_splits))
        
        # 
        split_files = self.files[self.files['split'] == split]

        voxelized_meshes = []
            
        labels = split_files['class'].unique().tolist()

        n = 0

        # Iterate through each class label
        for label in labels:
            files = split_files[split_files['class'] == label]
            files = files['object_path'].tolist()

            i = 0

            print(f'Voxelizing {split} meshes of the {label} class...')

            # Iterate through each .off file
            for file in files:

                # Get path to .off file
                if self.dataset == "ModelNet40":
                    path = self.path + 'ModelNet40/' + file
                ## Convert .npz file to .ply file and get path to new .ply file
                # elif self.dataset == "Toys4K":
                #     path = convert_toy4k_npz(self.path + file)
                else:
                    path = self.path + file
                
                # ShapeNetCore provides Binvox files, so we can skip the voxelization step
                if self.dataset == "ShapeNetCore.v2":
                    binvox_path = path
                # Otherwise, we need to voxelize the mesh
                else:
                    # Define arguments for cuda_voxelizer executable
                    args = ['./cuda_voxelizer', '-f', path, '-s', '32', '-o', 'binvox']

                    # Run the cuda_voxelizer executable to create a binvox file (This is orders of magnitude faster than using trimesh4.0)
                    popen = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    popen.wait()

                    # In case cuda_voxelizer fails, fall back to trimesh4.0
                    if popen.returncode != 0:
                        print(f"Cuda_voxelizer failed for {path}. Falling back to trimesh4.0.")
                        voxel_grid = voxelize_backup(path, self.system)
                        voxelized_meshes.append(voxel_grid)
                        i+=1
                        continue

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

                # Since the model is facing the ground, we need to rotate the voxel grid 180 degrees around the x-axis
                if self.dataset == "Toys4K":
                    voxel_grid = np.rot90(voxel_grid, k=4, axes=(0, 1))
                    voxel_grid = np.rot90(voxel_grid, k=1, axes=(1, 2))
                    # Right now the model is oriented properly, but the model is facing the wrong direction
                    # We need to flip the model along the y-axis
                    voxel_grid = np.flip(voxel_grid, axis=1)
                    

                # Add voxel grid to list of voxel grids
                voxelized_meshes.append(voxel_grid)

                # Render the voxel grid as a 3D plot
                if render:
                    render_voxel_grid(voxel_grid)

                i+=1

            n += 1
            print(f'Finished voxelizing {i} {split} meshes in the {label} class. ({n}/{len(labels)})')

        return voxelized_meshes

def normalize_0_1(data):
    """
    Normalize data in an array to be between 0 and 1 (in our case, 0 or 1)
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def voxelize_backup(path, system):
    """
    Fallback function for voxelize() in case cuda_voxelizer executable fails
    """
    binvox_path = ''
    if system == 'Unix':
        binvox_path = '../Dependencies/binvox'
    elif system == 'Windows':
        binvox_path = '../Dependencies/binvox.exe'
    elif system == 'OSX':
        binvox_path = '../Dependencies/binvox_osx'
    
    mesh = trimesh.load(path)

    voxel_grid = voxelize_binvox(mesh, dimension=32, binvox_path=binvox_path)

    # Extract the voxel grid as a numpy array
    voxel_grid = voxel_grid.matrix.astype(np.int8)

    # Normalize the voxel grid to be either 0 and 1
    voxel_grid = normalize_0_1(voxel_grid)

    return voxel_grid

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

def render_voxel_grid(data):
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

def save_voxels(data, labels=None, save_path='./Data/temp.npz', overwrite=False):
    i = 1
    
    # Create a new file name if the file already exists and overwrite is false
    while os.path.exists(save_path) and not overwrite:
        save_path = save_path[:-4] + f'_{i}' + save_path[-4:]
        i += 1

    print(f'Saving voxel grids to {save_path}...')
    np.savez_compressed(save_path, data=data, labels=labels)