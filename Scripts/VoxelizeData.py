import os
import trimesh
from trimesh.voxel.creation import voxelize_binvox
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

def normalize_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class Voxelize():
    def __init__(self, path, dataset, system='Unix'):

        allowed_datasets = ['ModelNet40', 'ShapeNetCore.v2', 'toys4k_point_clouds']

        allowed_OS = ['Unix', 'Windows']

        if dataset not in allowed_datasets:
            raise ValueError('Dataset not supported. Please choose from: {}'.format(allowed_datasets))
        
        if system not in allowed_OS:
            raise ValueError('OS not supported. Please choose from: {}'.format(allowed_OS))

        self.path = path
        self.dataset = dataset
        self.system = system
        self.files = self.get_files()
        self.train = self.voxelize('train')
        #self.test = self.voxelize('test')

        #self.render(self.train[0])
        #self.render(self.test[0])


    def get_files(self):

        if self.dataset == "ModelNet40":
            metadata = pd.read_csv(self.path + 'metadata_modelnet40.csv')
            metadata.drop(columns=['object_id'], inplace=True)
            print(metadata.head())
            return metadata

    def voxelize(self, split):
        
        allowed_splits = ['train', 'test']

        if split not in allowed_splits:
            raise ValueError('Split not supported. Please choose from: {}'.format(allowed_splits))
        
        if self.dataset == "ModelNet40":
            # 
            files = self.files[self.files['split'] == split]
            files = files['object_path'].tolist()

            voxelized_meshes = []

            i = 0

            # Iterate through each .off file
            for file in files:
                
                # Get path to .off file
                path = self.path + '/ModelNet40/' + file

                # Load .off file as a trimesh object
                mesh = trimesh.load(path)

                print(f'Loaded mesh {i} {file}')

                if self.system == 'Unix':
                    # Voxelize the mesh
                    voxels = voxelize_binvox(mesh, dimension=32, binvox_path='./binvox')
                elif self.system == 'Windows':
                    # Voxelize the mesh
                    voxels = voxelize_binvox(mesh, dimension=32, binvox_path='./binvox.exe')

                print(f'Voxelized mesh {i} {file}')

                # Extract the voxel grid as a numpy array
                #voxel_grid = voxels.matrix.astype(np.float32)
                voxel_grid = voxels.matrix.astype(np.float32)

                print('Extracted voxel grid')
                print(voxel_grid.shape)

                # Reshape the voxel grid to be the correct dimensions for the input layer of the network
                #voxel_grid = voxel_grid.reshape((32, 32, 32))

                # print('Reshaped voxel grid')

                # Normalize the voxel grid to be between 0 and 1
                voxel_grid = normalize_0_1(voxel_grid)

                # Add voxel grid to list of voxel grids
                voxelized_meshes.append(voxel_grid)

                print('Added voxel grid to list')

                self.render(voxel_grid)

                i+=1

            return voxelized_meshes
    

    def render(self, data):
        
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


