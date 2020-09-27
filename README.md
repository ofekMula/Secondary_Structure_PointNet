# Secondary_Structure_PointNet
protein secondary structure classification using Deep learning with PointNet architecture
<p float="left">
  <img src="https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/pointnet.JPG" width="400" />
  <img src="https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/1w5o_dspp.jpg" width="400" /> 
</p>


## Objective
In this project we deal with classification of amino acids in proteins into their secondary structurewith deep learning by using PointNet architecture.
The main objective of it was to examine whether PointNet architecture is good enough for that task, and also to compare between three different variants of it.
The problem of predicting the secondary structure of each amino acid is considered highly important in the structural biology field, due to its immense effects on the understanding of protein structure: the secondary structure provides a higher level of understanding of how a protein works. This can allow researchers to deduce how to affect, control and modify the functionality of a protein; and also lead to more precise results in related research (e.g. protein-protein docking).
Also has applications in drug development.

## Background on proteins
A protein is a molecule made of a folded chain of amino acids, linked together by peptide bonds. Those bonds are established between the carboxyl group of the amino acid and the amino group of the other acid. All proteins are constructed of the same building blocks, amino acids.
The function of a protein depends significantly on the strength of its structure, and this structure is obtained by the exact list of amino acids that construct it. Each protein folds in space into a three-dimensional shape, as a result of chemical bonds between the various amino acids.
Former researchers concluded that the protein structure can be divided into four levels : (1) Primary (2)  Secondary (3) Tertiary and (4) Quaternary.
We focused on our project on the Secondary structure.

![alt text](https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/%E2%80%8F%E2%80%8Fprotein_secondary_structure.JPG)

## PointCloud
PointNet architectures accept, as an input, a set of points in the Euclidean space R^3, called Point Cloud.
Each point is represented uniquely by its three coordinates (x,y,z) plus extra properties (such as color, normal etc.)
In this project each protein coverted to a pointcloud.
For example chain A of 1w3o protein:
<p float="left">
  <img src="https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/1w5o_dspp_2.jpg" width="400" />
  <img src="https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/1w5o_dspp.jpg" width="400" /> 
</p>

## PointNet Arcitecture
PointNet is a deep learning network of classification geometric 3D data. This network consumes a set of point clouds as an input.
The network has three key modules: the max pooling layer as a symmetric function to aggregate information from all the points, a local and global information combination structure, and two joint alignment networks that align both input points and point features. We summarize here some of the important features of the network (more details on them can be found in PointNet’s article):
Input - n points, each represented by its coordinates (x, y, z).
Output -
The classification task classifies the whole point cloud into one of k pre-determined classes. More precisely, it generates a score for each class (the “chance” of the input to be in that class).
The segmentation task, however, classifies each point in the cloud into one of m classes. 

## The models we examined

In our research, we compare three interesting methodologies for attacking the problem of secondary structure classification:
The basic model (PointNet8) is as described above - semantic segmentation of each residue into one of the 8 DSSP classes. 
The second model (PointNet3) works the same - but classify each point into one of the 3 main classes described above (helices, sheets and loops). Using this model, we hope to “ignore” the small differences between different types of DSSP types in the same “main class” (for example, the difference between a 4-turn helix and a 5-turn helix).
The last model (LocalPointNet3) takes a slightly different approach. Instead of using the entire protein as an input, we take a small neighborhood of each residue (for instance, the 16 closest residues). This neighborhood is normalized and then sent as an input to the classification network - and classified into one of the 3 main classes. The motivation for that model was described in the introduction for this paper - points in the same neighborhood are most likely to have a similar secondary structure.

## results

![alt text](https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/%E2%80%8F%E2%80%8Ftable%20results.JPG) 

visualization of the results:

![alt text](https://github.com/ofekMula/Secondary_Structure_PointNet/blob/final_project/Images/prediction_results.JPG) 

## future work
We propose here different suggestions for future improvements for our models.
1. Constructing the database from the backbone coordinates of each residue (we took only the carbon-alpha coordinates).

2. Regarding semantic segmentation into 8 classes, the database we used was biased to helix. Using a less biased database may improve accuracy. 

3. It would be useful to attack this problem (secondary structure classification) using a different neural network. It would be interesting to compare between the different architectures (for instance - DGCNN, PointNet++, PointConv, SpiderCNN, PointCNN).

## Credits
<a href="https://github.com/charlesq34/pointnet" target="_blank"> PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
Created by Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas from Stanford University.

## What we learned
