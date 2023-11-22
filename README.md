# Comparing the Performance of 3D Classification Models on Various Datasets

## Authors:
- Kyle Lukaszek
- Luke Janik-Jones

## Project Description:
3D classification models are neural networks that identify features in 3D meshes or point clouds to predict objects based on their characteristics. We plan on comparing different 3D object classification models that have been developed over the past few years using various common 3D object datasets, such as ModelNet40 and ShapeNet. We plan on using existing models as well as a simple CNN implementation that we will write so that we can evaluate how robust 3D classification models have become. We plan on measuring the performance of these models using accuracy as our main measure since Princeton recommends using accuracy for ModelNet and Stanford also recommends accuracy for ShapeNet. For any other datasets that we can get our hands on, such as Toy4K, we will be using accuracy as our performance measure to stay in line with our previous observations. These datasets may come in different datatypes and some models require specific datatypes (such as point clouds, or voxelized meshes) so we will have to find an efficient and effective way to convert these large datasets into our desired datatypes. Once we have trained and tested these models for each dataset, we will create plots for all of the data that we have collected so that we can easily compare the results. After that, we will write a detailed analysis and explain why some models outperformed others using our findings as evidence.

## Objectives & Sub-Objectives:
The main objective of this project is to determine the relative performance of 3D object classification models and see what differences between the models result in the highest performance and why. We also want to, if possible, implement a model of our own, and see how such a model could compare to the ones we chose. A sub-goal of ours would be to achieve similar accuracy in our model to some of the older and less accurate models.

## Proposed Methodology and Solution:
We intend to compare somewhere from three to six 3D classification models, including our own, on several datasets of 3D meshes and point clouds. In doing this, we will observe the achieved accuracy of these models to determine which models have a greater performance and why. Currently, we have several possible classification models to compare, including VoxNet, GCNN, RSCNN, and a simple CNN. We will compare the accuracy of these models over several datasets, such as ModelNet, ShapeNet, and Toy4K. Each of these datasets contains many three-dimensional objects for the models to categorize. After the algorithms have been implemented, we can plot their accuracies and assess their performances accordingly.

## Expected Outcome:
The expected result is that we’ll see similar relative performances of models between the different datasets that we are testing on. We also expect that our model will be significantly worse than the average 3D classification model, but potentially comparable to some of the oldest algorithms.

## Deliverables:
The deliverables we intend to present are a set of graphs or tables plotting the accuracy of these models across the different datasets so that we can illustrate the difference in performances between them. We also intend to deliver an analysis of why we believe each algorithm performed as they did, and what that implies for 3D classification in general. Lastly, we also intend to present our algorithm as well as an analysis of our decisions and results regarding its implementation.

## Work Division:
We will divide the work between us to ensure an equal amount of time is put into the project from both of us in both the implementation of the models and the analysis and writing. Since there may be overlap in implementation, we do not intend to divide the programming of the different models between us, instead, we intend to work collaboratively on them.


## Resources:
- Datasets (We have been granted access to all datasets):
- ModelNet10 & ModelNet40 (https://modelnet.cs.princeton.edu/#)
- ShapeNet (https://shapenet.org/)
- OmniObjects3D (https://omniobject3d.github.io/)
- Toys4K (https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k)
- CO3D Single Sequence Subset (https://github.com/facebookresearch/co3d)

	Libraries and Tools:
- Open3D (http://www.open3d.org/docs/release/getting_started.html#python)
- OpenCV (https://opencv.org/)
- PyTorch3D (https://pytorch3d.org/docs/renderer_getting_started)
- PyTorch
- Antiprism (https://www.antiprism.com/programs/off2obj.html)

## References:

Models Papers:
- ModelNet Datasets and Models To Use:
    - https://modelnet.cs.princeton.edu/ 
- Relation-Shape Convolutional Neural Network for Point Cloud Analysis
    - https://arxiv.org/abs/1904.07601
- VoxNet 3D CNN Real-Time Object Classifier
    - https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf
- Unsupervised Feature Learning for Point Cloud by Contrasting and Clustering With Graph Convolutional Neural Network
    - https://arxiv.org/abs/1904.12359
- 3D Model Classification Using Convolutional Neural Network
    - https://cs229.stanford.edu/proj2015/146_report.pdf
- Neural Network For 3D Object Classification:
    - http://cs231n.stanford.edu/reports/2016/pdfs/417_Report.pdf
- TriMesh2 (C++) Fork:
    https://github.com/Forceflow/trimesh2
- CudaVoxelizer:
    - https://github.com/Forceflow/cuda_voxelizer
- TriMesh4 (Python):
    - https://trimesh.org/
