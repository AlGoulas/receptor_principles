# Organizational principles of receptor distribution in the human brain
Uncover receptor principles of the human brain with data mining

# Description
The code performs a series of multivariate stats, dimensionaltiy reduction, machine learning (support vector regression) analysis, as well as feature selection (recursive feature elimination) to gain insights into the organization of 15 different types of receptors in the human brain.

Receptor density data are acquired via histological proceissing of human tissue (quantitative receptor autoradiography)

# Use
Create a virtual environment (e.g., with conda) with the specifications enlisted in requirements.txt. Download or clone this repository. If the virtual environment was created sucessfully, no further installations are required - the main .py file can be execute (DistillPrinciples.py).

To do so, you have to specifify the following folder and paths in the DistillPrinciples.py file:
1. Folder to store the results (figures and tables). E.g., 
```
results_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/receptor-principles/results")
```
2. Folder to load the necessary data (.npy files in the recept_data folder of this repository). E.g.,
```
data_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/receptor-principles/data")
```

# Data




