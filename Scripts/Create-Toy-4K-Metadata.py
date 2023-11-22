import os
import pandas as pd

"""
Author: Kyle Lukaszek

We need to iterate through a directory with the following structure:

-> toys4k_point_clouds
    -> [class name]
        -> [class name]_[number]
            -> pc10K.npz

We need to create a metadata .csv file with the following structure:

object_id, class, split, object_path
0, airplane, train, toys4k_point_clouds/airplane/airplane_0/pc10K.npz
1, airplane, train, toys4k_point_clouds/airplane/airplane_1/pc10K.npz
2, airplane, train, toys4k_point_clouds/airplane/airplane_2/pc10K.npz
...

We want to split the dataset into 80% train and 20% test
"""

# Define the path to the directory containing the point clouds
path = '3D-Classification/Data/toys4k_obj_files/'

# Define the path to the directory where the metadata file will be saved
save_path = path + 'metadata_toys4k.csv'

if os.path.exists(save_path):
    print('Metadata file already exists. Deleting...')
    os.remove(save_path)

# Define the classes
classes = os.listdir(path)

# Define the splits
splits = ['train', 'test']

# Create an empty dataframe

metadata = pd.DataFrame(columns=['object_id', 'class', 'split', 'object_path'])

# Iterate through each class folder and create the metadata
for i, class_ in enumerate(classes):

    print(f'{path + class_}')

    items = os.listdir(path + class_)

    # Split the items into train and test
    train = items[:int(len(items)*0.8)]
    test = items[int(len(items)*0.8):]

    # Iterate through each item and add it to the metadata dataframe
    for j, item in enumerate(items):
        if item in train:
            split = 'train'
        else:
            split = 'test'

        metadata.loc[len(metadata)] = {'object_id': j, 'class': class_, 'split': split, 'object_path': class_ + '/' + item + '/mesh.obj'}

# Save the metadata dataframe as a .csv file
metadata.to_csv(save_path, index=False)
