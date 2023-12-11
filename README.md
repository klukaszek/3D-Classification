# Comparing the Performance of 3D Classification Models on Various Datasets

## Authors:
- Kyle Lukaszek
- Luke Janik-Jones

## GitHub Repository:
https://github.com/klukaszek/3D-Classification

## Final Report:
The final report for this project is provided as a PDF file in the root directory. It is titled 'CIS4780 Final Report.pdf'. The report contains a detailed description of the project, the results, and the conclusions we drew from the results.

## Datasets:
The datasets we used for this project are expected to be in the subdirectories of the `data` directory. You will need the compressed versions of the datasets to use the models. Please contact us if you would like access to compressed npz files since we cannot upload it to GitHub.

## Testing Notebooks:

The notebooks we used for testing the models are in the root directory. Within each notebook, you will have to change a string in the second cell to select the dataset to use. The notebooks are as follows:

- 'OurCNN-Test.ipynb' - This notebook contains the code for training and testing the OurCNN model on a specified dataset.

- '3DCNN-Test.ipynb' - This notebook contains the code for training and testing the 3DCNN model on a specified dataset.

- 'VoxNet-Test.ipynb' - This notebook contains the code for training and testing the VoxNet model on a specified dataset.

## Scripts:
The scripts we used for this project are provided in the `scripts` directory. The scripts are as follows:
- 'VoxelizeData.py' - This script does all operations related to creating and handling the voxelized data. It takes in the raw data and outputs the voxelized data as .npz files. It also loads the voxelized data and provides functions to get the data in the correct format for the models.

- 'Create-Toy4K-Metadata.py' - This script takes in the directory structure of the Toy4K dataset and creates a metadata file that contains the class names, training and testing splits, and the file paths to the .obj files.

- 'Create-Shapenet-Metadata.ipynb' - This notebook takes in the directory structure of the ShapeNet dataset and creates a metadata file that contains the class names, training and testing splits, and the file paths to the .obj files. Basically the same as the Toy4K script but in a notebook.

- 'OurCNN.py' - This script contains the code for the OurCNN model.

- 'CNN.py' - This script contains the code for the 3DCNN model.

- 'VoxNet.py' - This script contains the code for the VoxNet model.