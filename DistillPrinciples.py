#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from random import randint

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import rankdata
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols

#sns.set(context='poster', style='white')

# Assume fonts are installed when exporting in .svg
plt.rcParams['svg.fonttype'] = 'none'

# Compute excitation/inhibition ratio
# ExcInh_Index: 1=Excitatory receptor 2=Inhibitory receptor
def compute_excinh(Data_I, Data_G, Data_S, ExcInh_Index):
    
    #Compute the sum of excitatory and inhitory receptors
    result = np.where(ExcInh_Index == 1)
    indexes_exc = result[0]
    
    result = np.where(ExcInh_Index == 2)
    indexes_inh = result[0]
    
    Exc_I = np.mean(Data_I[:,indexes_exc],1)
    Exc_G = np.mean(Data_G[:,indexes_exc],1)
    Exc_S = np.mean(Data_S[:,indexes_exc],1)
    
    # Granular dataset contains 0s for agranular areas - reassemble data
    # without this area
    result = np.where(Exc_G != 0)
    indexes_notzeros = result[0]
    Exc_G = Exc_G[indexes_notzeros]
    
    Inh_I = np.mean(Data_I[:,indexes_inh],1)
    Inh_G = np.mean(Data_G[:,indexes_inh],1)
    Inh_S = np.mean(Data_S[:,indexes_inh],1)
    
    Inh_G = Inh_G[indexes_notzeros]
    
    ExcInh_I = np.divide(Exc_I, Inh_I)
    ExcInh_G = np.divide(Exc_G, Inh_G)
    ExcInh_S = np.divide(Exc_S, Inh_S)
   
    return ExcInh_I, ExcInh_G, ExcInh_S, indexes_notzeros

# Compute excitation/inhibition ratio
# ReceptorTypes_IonoMetabo: 1=Ionotropic receptor 2=Metabotropic receptor
def compute_iono_metabo(Data_I, Data_G, Data_S, ReceptorTypes_IonoMetabo):
    #Compute the sum of excitatory and inhitory receptors
    result = np.where(ReceptorTypes_IonoMetabo == 1)
    indexes_iono = result[0]
    
    result = np.where(ReceptorTypes_IonoMetabo == 2)
    indexes_metabo = result[0]
    
    Iono_I = np.mean(Data_I[:,indexes_iono],1)
    Iono_G = np.mean(Data_G[:,indexes_iono],1)
    Iono_S = np.mean(Data_S[:,indexes_iono],1)
    
    Metabo_I = np.mean(Data_I[:,indexes_metabo],1)
    Metabo_G = np.mean(Data_G[:,indexes_metabo],1)
    Metabo_S = np.mean(Data_S[:,indexes_metabo],1)
    
    #Granular dataset contains 0s for agranular areas - reassemble data
    #without this area
    result = np.where(Iono_G != 0)
    indexes_notzeros = result[0]
    
    Iono_G = Iono_G[indexes_notzeros]
    
    #Use the same index since the same holds for the metabotropic summary
    Metabo_G = Metabo_G[indexes_notzeros]
   
    return Iono_I, Iono_G, Iono_S, Metabo_I, Metabo_G, Metabo_S, indexes_notzeros

# Biplot
def mybiplot(score, coeff, path_name_saved_file=None, score_labels=None, coeff_labels=None):
    scale_coeff = 0.9
    
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    xs = xs * scalex
    ys = ys * scaley
    
    #Plot scores
    fig = plt.figure()
    fig.set_size_inches(15, 15)  
    plt.scatter(xs, ys, s=200, marker='o', edgecolors='none')
    
    for i in range(len(xs)):
        plt.text(xs[i], ys[i], score_labels[i], fontsize=30)
    
    #Plot coefficients
    for i in range(n):
        plt.arrow(0, 0, scale_coeff*coeff[i,0], scale_coeff*coeff[i,1], color = 'r', alpha = 0.5)
        
        if coeff_labels is None:
            plt.text(scale_coeff*coeff[i,0]* 1.15, scale_coeff*coeff[i,1] * 1.15, "Var"+str(i+1), 
                     color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(scale_coeff*coeff[i,0]* 1.15, scale_coeff*coeff[i,1] * 1.15, coeff_labels[i], 
                     color = 'g', ha = 'center', va = 'center', fontsize=10)
            
    plt.grid()
    plt.savefig(path_name_saved_file, format="svg")

#Calculate entropy of each area across receptors
def calculate_entropy(ReceptData, normalize=True):
    
    # If normalization is requested then normalize each column of 
    # ReceptData by dividing with the max of each column. 
    # The ranges of receptor values vary so entropy could get 
    # influenced by such range differences
    
    # Get max of each column and divide each one by this max
    if(normalize==True): 
        
        max_column = np.max(ReceptData,0)
        ReceptData = ReceptData / max_column
  
    # Calculate the entropy for each area
    data_size=np.shape(ReceptData)
    H = np.zeros(data_size[0])
    
    for area in range(data_size[0]):
        
        current_receptor_profile = ReceptData[area,:]
        current_receptor_profile = np.divide(current_receptor_profile,
                                             np.sum(current_receptor_profile))
        
        H[area] = -np.sum(current_receptor_profile * np.log(current_receptor_profile)) \
        / np.log(len(current_receptor_profile));
    
    return H

# Plot and save scatterplot
def plot_save_scatter_plot(
        x, 
        y, 
        dataPoints_names=None, 
        path_name_saved_file=None, 
        title=None, 
        x_label=None, 
        y_label=None
    ): 
    
    data_to_plot = {'PC1':x, 'ReceptorDensity':y}
    df = pd.DataFrame(data_to_plot)

    fig = plt.figure()
    fig.set_size_inches(10, 10)  
      
    sns.regplot(x="PC1", y="ReceptorDensity", data=df,
            scatter_kws={"s":150,
                         "edgecolors":"none"}
            );

    for index, value in enumerate(y):     
        plt.text(x[index], value, dataPoints_names[index], fontsize=30)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(path_name_saved_file, format="svg")
       
def run_ancova(Y, X, Covariate, filename_results=None):
   
    data = {'Y':Y, 'X':X, 'Cov':Covariate}
    df = pd.DataFrame(data)

    formula = 'Y ~ X * C(Cov)'  # ANCOVA formula

    #Fit the model
    lm = ols(formula, df)
    fit = lm.fit()

    #Save the results
    print(fit.summary(), file=open(filename_results, "w")) 
    
    return fit 

def plot_rank_ordered_values(
        values, 
        labels=None, 
        path_name_saved_file=None, 
        title=None, 
        x_label=None, 
        y_label=None, 
        y_min=None, 
        y_max=None
    ):
    
    # Rank order the values and the rearrange the names accordingly
    sort_ind = np.argsort(values)
    values = values[sort_ind]
    labels = [labels[i] for i in sort_ind]
    
    # Plot
    data = {'X':values, 'AreaNames':labels}
    df = pd.DataFrame(data)
    
    fig = plt.figure()
    fig.set_size_inches(10, 10) 
    
    plot = sns.barplot(y = 'X', 
                       x = 'AreaNames',
                       color = 'magenta', 
                       data=df)
    
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plot.set(ylim=(y_min, y_max))
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Save figure in the spacified path with the specified name
    plt.savefig(path_name_saved_file, format="svg")
    
def custom_RFE(X, Y, test_size_perc=0.2, iterations=100, feature_names=None):
    svr = SVR(kernel="linear", C=1.0)
    # Initialize variables to keep the relevant info
    size_X = X.shape
    total_rfe_steps = size_X[1]-1
    MSE_of_rfe_step = np.array([[0.]*total_rfe_steps]*iterations)
    
    FeatureNames_RFE_steps=[]
    FeatureScores_RFE_steps=[]

    # This object will help us keep track of the positions by offering the 
    # indexes of the train and test set
    # sp = ShuffleSplit(n_splits=1, test_size=test_size_perc)
    
    size_Y = Y.shape
    Mean_AllPredictions = np.asarray([[0.] * total_rfe_steps] * size_Y[0])
    
    for rfe_steps in range(total_rfe_steps):
        size_X = X.shape
        features = size_X[1] - 1
        all_feature_selected = np.array([0]*size_X[1])
        
        all_mse = np.array([0.]*iterations)
        
        rfe = RFE(estimator=svr, n_features_to_select=features, step=1, verbose=0)
        AllPredictions = np.asarray([[0.] * iterations] * size_Y[0])
        
        for iter in range(iterations):
            index = np.random.permutation(size_Y[0])
            
            split_position = np.round(test_size_perc * size_Y[0])
            
            train_index = index[int(split_position):]  
            test_index = index[:int(split_position)]  

            X_train, X_test = X[train_index,:], X[test_index,:] 
            Y_train, Y_test = Y[train_index], Y[test_index]
                
            mean_X_train = np.mean(X_train, axis=0)
            std_X_train = np.std(X_train, axis=0)
            
            X_train_zscored = (X_train - mean_X_train) / std_X_train
            
            # Use the mean and std of the training set to scale the test set
            X_test_zscored = (X_test - mean_X_train) / std_X_train
            
            # Compute the mean and std for the target variable 
            # standardize train and test Y
            mean_Y_train = np.mean(Y_train)
            std_Y_train = np.std(Y_train)
    
            Y_train_zscored = (Y_train - mean_Y_train) / std_Y_train
            Y_test_zscored = (Y_test - mean_Y_train) / std_Y_train
            
            # Fit the model and predict from the test set
            rfe.fit(X_train_zscored, Y_train_zscored)
            Y_predicted = rfe.predict(X_test_zscored)
            
            # Store the mse
            all_mse[iter] = mean_squared_error(Y_test_zscored, Y_predicted)
            
            AllPredictions[test_index, iter] = Y_predicted
                  
            # Mark the features that are selcted in this iteration 
            # (marked with ones)
            feature_selected = rfe.ranking_
            feature_selected[feature_selected > 1] = 0
            
            # This vector contains the features that are ranked the highest
            # Hence the higher these values the most consistent the feature 
            # selection
            all_feature_selected = all_feature_selected + feature_selected 
            
            
        # Store the mean mse across all iternal iterations for this rfe step    
        # MSE_of_rfe_step[rfe_steps] = np.mean(all_mse)
        
        MSE_of_rfe_step[:, rfe_steps] = all_mse
        
        # Here we have to summarize all the predictions by averaging the 
        # predictions across the terations BUT ONLY by taking into account the 
        # times that each datapoint of the target variable was taken into 
        # account
        
        sum_AllPredictions = np.sum(AllPredictions, axis=1)
        ind = np.where(AllPredictions != 0)
        
        AllPredictions[ind] = 1
        timesiterated_AllPredictions = np.sum(AllPredictions, axis=1)
        
        Mean_AllPredictions[:, rfe_steps] = sum_AllPredictions / timesiterated_AllPredictions
        
        # Reduce the feature matrix by discarding the feature with the smallest 
        # consistency score
        indexes_min = np.where(all_feature_selected == all_feature_selected.min())
        indexes_min = indexes_min[0]
        
        # In case of more than one feature with minimum value select
        # one of these features at random
        if(len(indexes_min) > 1):
           indexes_min = indexes_min[randint(0, len(indexes_min)-1)]
        
        all_feature_selected[indexes_min] = -1
        indexes_shrink = np.where(all_feature_selected > -1)
        indexes_shrink = indexes_shrink[0]
        
        # Shrink the feature matrix by excluding the feature with the lowest 
        # score
        X = X[:,indexes_shrink]
        
        # Shrink the names of the features
        res_list = [feature_names[i] for i in indexes_shrink]
        
        FeatureNames_RFE_steps.append(res_list)
        FeatureScores_RFE_steps.append(all_feature_selected[indexes_shrink])

        #Update here the feature names list
        feature_names = res_list[:]

    return MSE_of_rfe_step, FeatureNames_RFE_steps, FeatureScores_RFE_steps, Mean_AllPredictions 

def show_img(x, title=None, xlabel=None, ylabel=None, x_tick_labels=None):
    plt.figure()
    plt.imshow(x, extent=[0,5,0,5])
    plt.colorbar()
    
    ax = plt.gca()
    ax.set_xticklabels(x_tick_labels)
    plt.xticks(rotation=90)
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def read_area_colors(path_to_file):
    # Open the file and read it line by line
    # Extract the area name, RGB color and convert the latter to HEX
    f = open(path_to_file, "r")
    area_colors = {}
    for line in f:
        s = line.split(';')#first string is area name second are the seperated (with ',') RGB values
        rgb = s[1].split(',')#rgb contains the seperate RGB values [0 255]
        hex_value = matplotlib.colors.to_hex([int(rgb[0])/255,  
                                              int(rgb[1])/255,  
                                              int(rgb[2])/255
                                             ]
                                            )
        area_colors[s[0]] = hex_value
    f.close()     
      
    return area_colors 

def read_replace_write(path_to_file, 
                       path_to_new=None, 
                       filename_new=None,
                       list_to_replace=None,
                       list_new=None
                       ):
    # Open the file and read it line by line
    # Extract the area name, RGB color and convert the latter to HEX
    f_read = open(path_to_file, "r")
    path_to_save = path_to_new / filename_new
    f_write = open(path_to_save, "w")
    
    # Read the file line by line and check each time if any item from the 
    # list_to_replace is present
    for line in f_read:
        is_in_list = [v in line for v in list_to_replace]# check if list_to_replace are in line 
        if any(is_in_list):
            idx = [i for i,item in enumerate(is_in_list) if item==True]#get the list_to_replace of items found in line
            for i in idx: 
                line = line.replace(list_to_replace[i], list_new[i]) 
        f_write.write(line)    
        
    f_read.close() 
    f_write.close() 

def values_to_colormap_hex(area_value, 
                           colormap=cm.viridis, 
                           min_val=None, 
                           max_val=None):
    '''
    Given the values in values, return a list of HEX colors representing
    the color-tranformed values based on the provided colormap.
    
    Input
    -----
    area_value: dict, key:area names, str value: value of area, float
    
    colormap: matplotlib.colors.ListedColormap object, default cm.viridis
    
    min_val: float, min value for constructing the colormap, 
        default None, hence internally set to: min([*area_value.values()])
        
    max_val: float, max value for constructing the colormap, 
        default None, hence internally set to: max([*area_value.values()]) 
        
    Output
    ------
    area_color: dict, 
        key: str, area names, same as area_value input arg
        value: str, value of area tranformed to HEX color      
    
    '''
    # Set min max if not specified 
    if min_val is None: min_val = min([*area_value.values()])
    if max_val is None: max_val = max([*area_value.values()])
    
    norm = matplotlib.colors.Normalize(vmin = min_val, 
                                       vmax = max_val)
    cmap = colormap
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Create a dict with area as key and the HEX value-to-colormap as value 
    area_color={}
    for key,value in zip(area_value.keys(), area_value.values()): 
        rgba = m.to_rgba(value)
        area_color[key] = matplotlib.colors.to_hex(rgba)
    
    return area_color

def create_transformation_list(area_map_values, area_color_values_to_map=None):
    '''
    Given two dictionaries area_map_values, area_color_values_to_map, create 
    two lists old_values, new_values with 1:1 correspondance specifying an old
    and new values, such that new_values contains the values of 
    area_new_color_values corresponding to keys that are also found in
    area_map_values.
    
    Input
    -----
    area_map_values: dict, 
        key: str, area names, same as area_value input arg
        value: str, value of area tranformed to HEX color 
        
    area_color_values_to_map: same as area_map_values

    NOTE: len(area_map_values) not necessary len(area_color_values_to_map)

    Output
    ------
    old_values: list, values in area_map_values that correspond to existing 
        keys in area_map_values, a search corresponding to an iteration of keys 
        in area_color_values_to_map
        
    new_values: list, values in area_color_values_to_map that correspond to 
        area_color_values_to_map keys found in area_map_values keys 
    
    '''
    old_values = []
    new_values = []
    for key,value in zip(area_color_values_to_map.keys(), 
                         area_color_values_to_map.values()):
        if key in area_map_values:
            old_values.append(area_map_values[key])
            new_values.append(value)
    
    return old_values, new_values  
    
# Path to save results - individual names of files will be appended to this path
results_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/development/receptor-principles/results")

# Path with the data in .npy format
data_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/development/receptor-principles/data")

# Load all the necessary data
file_to_open = data_folder / "ReceptData_I.npy"
ReceptData_I = np.load(file_to_open)

file_to_open = data_folder / "ReceptData_G.npy"
ReceptData_G = np.load(file_to_open)

file_to_open = data_folder / "ReceptData_S.npy"
ReceptData_S = np.load(file_to_open)

file_to_open = data_folder / "ReceptorTypes_ExcInh.npy"
ReceptorTypes_ExcInh = np.load(file_to_open)

file_to_open = data_folder / "RegionNames.npy"
RegionNames = np.load(file_to_open)

file_to_open = data_folder / "ReceptorNames.npy"
ReceptorNames = np.load(file_to_open)

file_to_open = data_folder / "ReceptorTypes_IonoMetabo.npy"
ReceptorTypes_IonoMetabo = np.load(file_to_open)

file_to_open = data_folder / "G1_BigBrain.npy"
G1_BigBrain = np.load(file_to_open)

# Load RGB valeus for each region - used for visualization
area_colors_file = "/Users/alexandrosgoulas/Data/work-stuff/python-code/packages/receptor_principles/visualization/areas_RGB.txt"
map_file_lateral = "/Users/alexandrosgoulas/Data/work-stuff/python-code/packages/receptor_principles/visualization/human_map_lateral.svg" 
map_file_medial = "/Users/alexandrosgoulas/Data/work-stuff/python-code/packages/receptor_principles/visualization/human_map_medial.svg" 

# Load the HEX color and the area names in a dictionary (key: names, values: HEX)
areas_hex = read_area_colors(area_colors_file)

# Analyze the data     

# Calculate excitation inhibition 
(ExcInh_I, ExcInh_G, ExcInh_S, 
 indexes_notzeros) = compute_excinh(ReceptData_I, 
                                    ReceptData_G, 
                                    ReceptData_S, 
                                    ReceptorTypes_ExcInh)

# Keep only the data with values - hence remove agranular areas
ReceptData_Reduced_I = ReceptData_I[indexes_notzeros, :]
ReceptData_Reduced_G = ReceptData_G[indexes_notzeros, :]
ReceptData_Reduced_S = ReceptData_S[indexes_notzeros, :]

ExcInh_I = ExcInh_I[indexes_notzeros]
ExcInh_S = ExcInh_S[indexes_notzeros]

RegionNames_Reduced = np.array(RegionNames)[indexes_notzeros]

# Visualize ExcInh_I, ExcInh_G, ExcInh_S in the surface
All_Values = [ExcInh_I, ExcInh_G, ExcInh_S]
All_Acronyms = ['ExcInh_I', 'ExcInh_G', 'ExcInh_S']

for excinh, acronym in zip(All_Values, All_Acronyms):
    # Dict with area names as keys and PC1 values as values
    areas_values = dict(zip(RegionNames_Reduced, excinh)) 
    
    # Create dict with names as keys and values-to-hexcolor as values
    area_color_hex = values_to_colormap_hex(areas_values, 
                                            colormap=cm.viridis
                                            )
    
    # Create old and new corresponding lists of HEX colors 
    # new_list contains the HEX colors to be painted in the svg file
    old_list, new_list = create_transformation_list(areas_hex, 
                                                    area_color_values_to_map=area_color_hex)
    
    read_replace_write(map_file_lateral, 
                       path_to_new=results_folder, 
                       filename_new= acronym + '_map_lateral.svg',
                       list_to_replace=old_list,
                       list_new=new_list
                       )
    
    read_replace_write(map_file_medial, 
                       path_to_new=results_folder, 
                       filename_new= acronym + '_map_medial.svg',
                       list_to_replace=old_list,
                       list_new=new_list
                       )

# Plot rank ordered exc/inh for each region and laminar compartment
file_to_save = results_folder / "ExcInh_RankOrdered_I.svg"

plot_rank_ordered_values(ExcInh_I, 
                         labels=RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Exc/Inh Infragranular", 
                         x_label="",
                         y_label="Exc/Inh", 
                         y_min=np.min(ExcInh_I)-0.1, 
                         y_max=np.max(ExcInh_I)
                         )

file_to_save = results_folder / "ExcInh_RankOrdered_G.svg"

plot_rank_ordered_values(ExcInh_G, 
                         labels=RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Exc/Inh Granular", 
                         x_label="", 
                         y_label="Exc/Inh", 
                         y_min=np.min(ExcInh_G)-0.1, 
                         y_max=np.max(ExcInh_G)
                         )

file_to_save = results_folder / "ExcInh_RankOrdered_S.svg"

plot_rank_ordered_values(ExcInh_S, 
                         labels=RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Exc/Inh Supragranular", 
                         x_label="", 
                         y_label="Exc/Inh", 
                         y_min=np.min(ExcInh_S)-0.1, 
                         y_max=np.max(ExcInh_S)
                         )

# Calculate Entropy

# Normalize densities per receptor
max_ReceptData_Reduced_I = np.max(ReceptData_Reduced_I, axis=0)
max_ReceptData_Reduced_G = np.max(ReceptData_Reduced_G, axis=0)
max_ReceptData_Reduced_S = np.max(ReceptData_Reduced_S, axis=0)

ReceptData_Reduced_I_norm = ReceptData_Reduced_I / max_ReceptData_Reduced_I
ReceptData_Reduced_G_norm = ReceptData_Reduced_G / max_ReceptData_Reduced_G
ReceptData_Reduced_S_norm = ReceptData_Reduced_S / max_ReceptData_Reduced_S

H_I = calculate_entropy(ReceptData_Reduced_I_norm)
H_G = calculate_entropy(ReceptData_Reduced_G_norm)
H_S = calculate_entropy(ReceptData_Reduced_S_norm)

file_to_save = results_folder / "H_RankOrdered_I.svg"

plot_rank_ordered_values(H_I, 
                         labels=RegionNames_Reduced, 
                         path_name_saved_file=file_to_save,
                         title="Rank ordered regions Entropy Infragranular", 
                         x_label="", 
                         y_label="Entropy", 
                         y_min=np.min(H_I)-0.01, 
                         y_max=np.max(H_I)
                         )

file_to_save = results_folder / "H_RankOrdered_G.svg"

plot_rank_ordered_values(H_G, 
                         labels=RegionNames_Reduced, 
                         path_name_saved_file=file_to_save,
                         title="Rank ordered regions Entropy Granular", 
                         x_label="", 
                         y_label="Entropy", 
                         y_min=np.min(H_G)-0.01, 
                         y_max=np.max(H_G)
                         )

file_to_save = results_folder / "H_RankOrdered_S.svg"

plot_rank_ordered_values(H_S, 
                         labels=RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Entropy Supragranular", 
                         x_label="", 
                         y_label="Entropy", 
                         y_min=np.min(H_S)-0.01, 
                         y_max=np.max(H_S)
                         )

# Visualize H_I, H_G, H_S in the surface
All_Values = [H_I, H_G, H_S]
All_Acronyms = ['H_I', 'H_G', 'H_S']

for excinh, acronym in zip(All_Values, All_Acronyms):
    # Dict with area names as keys and PC1 values as values
    areas_values = dict(zip(RegionNames_Reduced, excinh)) 
    
    # Create dict with names as keys and values-to-hexcolor as values
    area_color_hex = values_to_colormap_hex(areas_values, 
                                            colormap=cm.viridis
                                            )
    
    # Create old and new corresponding lists of HEX colors 
    # new_list contains the HEX colors to be painted in the svg file
    old_list, new_list = create_transformation_list(areas_hex, 
                                                    area_color_values_to_map=area_color_hex)
    
    read_replace_write(map_file_lateral, 
                       path_to_new=results_folder, 
                       filename_new= acronym + '_map_lateral.svg',
                       list_to_replace=old_list,
                       list_new=new_list
                       )
    
    read_replace_write(map_file_medial, 
                       path_to_new=results_folder, 
                       filename_new= acronym + '_map_medial.svg',
                       list_to_replace=old_list,
                       list_new=new_list
                       )

# Create a list of receptor names with a prefix indicating the layer 
# (for PCA visualization)
ReceptorNames_I_G_S = [ ]
for i in range(len(ReceptorNames)*3):   
    if i >= 0 and i <= 14:
        ReceptorNames_I_G_S.append(ReceptorNames[i] + '_I')
    if i > 14 and i <= 29:
        ReceptorNames_I_G_S.append(ReceptorNames[i-15] + '_G')
    if i > 29 and i <= 44:
        ReceptorNames_I_G_S.append(ReceptorNames[i-30] + '_S')

# Run PCA on the receptor data after they are z-scored
X = np.concatenate((ReceptData_Reduced_I, ReceptData_Reduced_G, ReceptData_Reduced_S), 
                   axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
scores = pca.fit_transform(X_scaled)
coeff = np.transpose(pca.components_[0:2, :])# Do we have to transpose? Yes!

PC1 = scores[:,0] 
PC1_ranked = rankdata(PC1) 

# Flip the sign for visualization purposes
PC1_2 = scores[:, 0:2]
PC1_2[:,1] = -1*PC1_2[:, 1]
coeff[:,1] = -1*coeff[:, 1] 

file_to_save = results_folder / "biplot.svg"

mybiplot(PC1_2, 
         coeff, 
         path_name_saved_file=file_to_save, 
         score_labels=RegionNames_Reduced, 
         coeff_labels=ReceptorNames_I_G_S
         )

# Visualize PC1 in flat maps

# Dict with area names as keys and PC1 values as values
areas_PC1 = dict(zip(RegionNames_Reduced, PC1)) 

# Create dict with names as keys and values-to-hexcolor as values
area_color_hex = values_to_colormap_hex(areas_PC1, 
                                        colormap=cm.viridis
                                        )

# Create old and new corresponding lists of HEX colors 
# new_list contains the HEX colors to be painted in the svg file
old_list, new_list = create_transformation_list(areas_hex, 
                                                area_color_values_to_map=area_color_hex)

read_replace_write(map_file_lateral, 
                   path_to_new=results_folder, 
                   filename_new='PC1_map_lateral.svg',
                   list_to_replace=old_list,
                   list_new=new_list
                   )

read_replace_write(map_file_medial, 
                   path_to_new=results_folder, 
                   filename_new='PC1_map_medial.svg',
                   list_to_replace=old_list,
                   list_new=new_list
                   )

# Calculate the correlation between ExcInh across PC1 and plot the data
rho_ExcInh_S, pval_ExcInh_S = spearmanr(PC1, ExcInh_S)
rho_ExcInh_G, pval_ExcInh_G = spearmanr(PC1, ExcInh_G)  
rho_ExcInh_I, pval_ExcInh_I = spearmanr(PC1, ExcInh_I)
    
# Plot the relation between Exc/Inh and PC1

# Infragranular layers
file_to_save = results_folder / "ExcInh_I.svg"

plot_save_scatter_plot(PC1, 
                       ExcInh_I, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save,
                       title="Infragranular Layers", 
                       x_label="PC1", 
                       y_label="Exc/Inh receptor density"
                       )

# Granular layers
file_to_save = results_folder / "ExcInh_G.svg"

plot_save_scatter_plot(PC1,
                       ExcInh_G, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Granular Layers", 
                       x_label="PC1", 
                       y_label="Exc/Inh receptor density"
                       )

# Supragranular layers
file_to_save = results_folder / "ExcInh_S.svg"

plot_save_scatter_plot(PC1, 
                       ExcInh_S, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Supragranular Layers", 
                       x_label="PC1", 
                       y_label="Exc/Inh receptor density"
                       )

# Run an ANCOVA model to test if the Exc/Inh and PC1 slopes are layer specific
ConcPC1_ranked = np.concatenate((PC1_ranked, PC1_ranked, PC1_ranked), 
                                axis=0)


ExcInh = np.concatenate((ExcInh_I, ExcInh_G, ExcInh_S), 
                        axis=0)
ExcInh_ranked = rankdata(ExcInh)

# Make a categorical predictor indicating what is supra=1 granular=2 infra=3
supra_index = np.asarray([3]*ExcInh_I.size)
granular_index = np.asarray([2]*ExcInh_I.size)
infra_index = np.asarray([1]*ExcInh_I.size)

Layer = np.concatenate((infra_index, granular_index, supra_index), 
                              axis=0)

file_to_save = results_folder / "summary_fit_ExcInh.txt" 

fit_ExcInh = run_ancova(ConcPC1_ranked,
                        ExcInh_ranked, 
                        Layer, 
                        filename_results=file_to_save
                        )

# Plot and save a boxplot for a summary of overall Exc/Inh in each layer
# Use unranked data for better interpertabilty
data_ExcInh_LayerWise = {'PC1':ConcPC1_ranked, 'ExcInh':ExcInh, 'Layer':Layer}
df = pd.DataFrame(data_ExcInh_LayerWise)

fig = plt.figure()
fig.set_size_inches(10, 10)  
sns.boxplot(x="Layer", y="ExcInh", data=df, palette="Set3")

file_to_save = results_folder / "ExcInh_LayerWise.svg"

plt.savefig(file_to_save, format="svg")

# Compute the correaltion between entropy of receptor density and PC1
rho_H_S, pval_H_S = spearmanr(PC1, H_S)
rho_H_I, pval_H_I = spearmanr(PC1, H_I)
rho_H_G, pval_H_G = spearmanr(PC1, H_G)

#Plot the correlation between Entropy across PC1 

#Infragranular layers
file_to_save = results_folder / "H_I.svg"

plot_save_scatter_plot(PC1, 
                       H_I, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Infragranular Layers", 
                       x_label="PC1", 
                       y_label="Entropy of receptor density"
                       )

#Granular layers
file_to_save = results_folder / "H_G.svg"

plot_save_scatter_plot(PC1, 
                       H_G, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Granular Layers", 
                       x_label="PC1", 
                       y_label="Entropy of receptor density"
                       )

#Supragranular layers
file_to_save = results_folder / "H_S.svg"

plot_save_scatter_plot(PC1, 
                       H_S, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Supragranular Layers", 
                       x_label="PC1", 
                       y_label="Entropy of receptor density"
                       )
    

# Fit an ANCOVA model to uncover if the relation of PC1 and Entropy of receptor
# density is meadiated by the Layer type
H = np.concatenate((H_I, H_G, H_S), axis=0)
H_ranked = rankdata(H)

# Make a categorical predictor indicating what is supra=1 granular=2 infra=3
supra_index = np.asarray([3]*H_I.size)
granular_index = np.asarray([2]*H_I.size)
infra_index = np.asarray([1]*H_I.size)

Layer = np.concatenate((infra_index, granular_index, supra_index), 
                              axis=0)

file_to_save = results_folder / "summary_fit_H.txt"

fit_H = run_ancova(ConcPC1_ranked, 
                   H_ranked, 
                   Layer, 
                   filename_results=file_to_save
                   )

# Plot and save a boxplot for a summary of overall Entropy in each laminae
# Use unranked data for better interpertabilty
data_H_LayerWise = {'PC1':ConcPC1_ranked, 'H':H, 'Layer':Layer}
df = pd.DataFrame(data_H_LayerWise)

fig = plt.figure()
fig.set_size_inches(10, 10)  
sns.boxplot(x="Layer", y="H", data=df, palette="Set3")

file_to_save = results_folder / "H_LayerWise.svg"

plt.savefig(file_to_save, format="svg") 

# Estimate overall density of ionotropic and metabotropic receptors and how
# they relate to PC1
(Iono_I, Iono_G, Iono_S, 
 Metabo_I, Metabo_G, Metabo_S, 
 indexes_notzeros_ionometabo) = compute_iono_metabo(ReceptData_I,
                                                  ReceptData_G,
                                                  ReceptData_S,
                                                  ReceptorTypes_IonoMetabo)

Iono_I = Iono_I[indexes_notzeros_ionometabo]
Iono_S = Iono_S[indexes_notzeros_ionometabo]           

Metabo_I = Metabo_I[indexes_notzeros_ionometabo]
Metabo_S = Metabo_S[indexes_notzeros_ionometabo]  

# Visualize Iono Metabo I, G, S in the surface
All_Values = [Iono_I, Iono_G, Iono_S, Metabo_I, Metabo_G, Metabo_S]
All_Acronyms = ['Iono_I', 'Iono_G', 'Iono_S', 'Metabo_I', 'Metabo_G', 'Metabo_S']

for excinh, acronym in zip(All_Values, All_Acronyms):
    # Dict with area names as keys and PC1 values as values
    areas_values = dict(zip(RegionNames_Reduced, excinh)) 
    
    # Create dict with names as keys and values-to-hexcolor as values
    area_color_hex = values_to_colormap_hex(areas_values, 
                                            colormap=cm.viridis
                                            )
    
    # Create old and new corresponding lists of HEX colors 
    # new_list contains the HEX colors to be painted in the svg file
    old_list, new_list = create_transformation_list(areas_hex, 
                                                    area_color_values_to_map=area_color_hex)
    
    read_replace_write(map_file_lateral, 
                       path_to_new=results_folder, 
                       filename_new= acronym + '_map_lateral.svg',
                       list_to_replace=old_list,
                       list_new=new_list
                       )
    
    read_replace_write(map_file_medial, 
                       path_to_new=results_folder, 
                       filename_new= acronym + '_map_medial.svg',
                       list_to_replace=old_list,
                       list_new=new_list
                       )


# Make a categorical predictor indicating that iono =1 and metabo =2
ConcPC1_ranked = np.concatenate((PC1_ranked, PC1_ranked), 
                                axis=0)

# This categorical predictor will be used for all I, G, S models since
# the size and arrangement of the predictors is identical
index_iono = np.asarray([1]*Iono_I.size)
index_metabo = np.asarray([2]*Metabo_I.size)

ReceptorType = np.concatenate((index_iono, index_metabo), 
                              axis=0)

IonoMetaboDensity_I = np.concatenate((Iono_I, Metabo_I), 
                                     axis=0)
IonoMetaboDensity_I_ranked = rankdata(IonoMetaboDensity_I)

IonoMetaboDensity_G = np.concatenate((Iono_G, Metabo_G), 
                                     axis=0)
IonoMetaboDensity_G_ranked = rankdata(IonoMetaboDensity_G)

IonoMetaboDensity_S = np.concatenate((Iono_S, Metabo_S), 
                                     axis=0)
IonoMetaboDensity_S_ranked = rankdata(IonoMetaboDensity_S)

# Plot ranked-order metabo and iono densities for each area in a layer-wise way.
# Iono
file_to_save = results_folder / "Iono_RankOrdered_I.svg"

plot_rank_ordered_values(Iono_I, 
                         RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Iono Infragranular", 
                         x_label="", 
                         y_label="Iono I", 
                         y_min=np.min(Iono_I)-50, 
                         y_max=np.max(Iono_I)
                         )

file_to_save = results_folder / "Iono_RankOrdered_G.svg"

plot_rank_ordered_values(Iono_G, 
                         RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Iono Granular",
                         x_label="", 
                         y_label="Iono G", 
                         y_min=np.min(Iono_G)-50, 
                         y_max=np.max(Iono_G)
                         )

file_to_save = results_folder / "Iono_RankOrdered_S.svg"

plot_rank_ordered_values(Iono_S, 
                         RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Iono Supragranular", 
                         x_label="", 
                         y_label="Iono S", 
                         y_min=np.min(Iono_S)-50, 
                         y_max=np.max(Iono_S)
                         )

# Metabo
file_to_save = results_folder / "Metabo_RankOrdered_I.svg"

plot_rank_ordered_values(Metabo_I, 
                         RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Metabo Infragranular", 
                         x_label="", 
                         y_label="Metabo I", 
                         y_min=np.min(Metabo_I)-50, 
                         y_max=np.max(Metabo_I)
                         )

file_to_save = results_folder / "Metabo_RankOrdered_G.svg"

plot_rank_ordered_values(Metabo_G, 
                         RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Metabo Granular", 
                         x_label="", 
                         y_label="Metabo G", 
                         y_min=np.min(Metabo_G)-50, 
                         y_max=np.max(Metabo_G)
                         )

file_to_save = results_folder / "Metabo_RankOrdered_S.svg"

plot_rank_ordered_values(Metabo_S, 
                         RegionNames_Reduced, 
                         path_name_saved_file=file_to_save, 
                         title="Rank ordered regions Metabo Supragranular", 
                         x_label="", 
                         y_label="Metabo S", 
                         y_min=np.min(Metabo_S)-50, 
                         y_max=np.max(Metabo_S)
                         )

# Plot boxplots for the iono and metabo receptors on a layer-wise manner
# Iono
Iono = np.concatenate((Iono_I, Iono_G, Iono_S), 
                      axis=0)
data_Iono_LayerWise = {'Iono':Iono, 'Layer':Layer}
df = pd.DataFrame(data_Iono_LayerWise)

fig = plt.figure()
fig.set_size_inches(10, 10)  
sns.boxplot(x="Layer", y="Iono", data=df, palette="Set3")

file_to_save = results_folder / "Iono_LayerWise.svg"
plt.savefig(file_to_save, format="svg")

#Metabo
Metabo = np.concatenate((Metabo_I, Metabo_G, Metabo_S), 
                        axis=0)
data_Metabo_LayerWise = {'Metabo':Metabo, 'Layer':Layer}
df = pd.DataFrame(data_Metabo_LayerWise)

fig = plt.figure()
fig.set_size_inches(10, 10)  
sns.boxplot(x="Layer", y="Metabo", data=df, palette="Set3")

file_to_save = results_folder / "Metabo_LayerWise.svg"
plt.savefig(file_to_save, format="svg")

#Fit the ancova models and save the results

file_to_save = results_folder / "summary_fit_IonoMetabo_I.txt"

fit_IonoMetaboDensity_I = run_ancova(ConcPC1_ranked, 
                                     IonoMetaboDensity_I_ranked, 
                                     ReceptorType, 
                                     filename_results=file_to_save
                                     )

file_to_save = results_folder / "summary_fit_IonoMetabo_G.txt"

fit_IonoMetaboDensity_G = run_ancova(ConcPC1_ranked, 
                                     IonoMetaboDensity_G_ranked, 
                                     ReceptorType, 
                                     filename_results=file_to_save
                                     )

file_to_save = results_folder / "summary_fit_IonoMetabo_S.txt"

fit_IonoMetaboDensity_S = run_ancova(ConcPC1_ranked, 
                                     IonoMetaboDensity_S_ranked, 
                                     ReceptorType, 
                                     filename_results=file_to_save
                                     )

# Plot seperately the relation of receptor density and PC1 for iono and metabo
# receptors
                       
#Supragranular layers - Ionotropic
file_to_save = results_folder / "Iono_S.svg"

plot_save_scatter_plot(PC1, 
                       Iono_S, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Supragranular Layers - Ionotropic", 
                       x_label="PC1", 
                       y_label="Receptor Density"
                       )

#Supragranular layers - Metabotropic
file_to_save = results_folder / "Metabo_S.svg"

plot_save_scatter_plot(PC1, 
                       Metabo_S, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Supragranular Layers - Metabotropic", 
                       x_label="PC1", 
                       y_label="Receptor Density"
                       )

#Granular layers - Ionotropic
file_to_save = results_folder / "Iono_G.svg"

plot_save_scatter_plot(PC1, 
                       Iono_G, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Granular Layers - Ionotropic", 
                       x_label="PC1", 
                       y_label="Receptor Density"
                       )

# Granular layers - Metabotropic
file_to_save = results_folder / "Metabo_G.svg"

plot_save_scatter_plot(PC1, 
                       Metabo_G, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Granular Layers - Metabotropic", 
                       x_label="PC1", 
                       y_label="Receptor Density"
                       )

# Infragranular layers - Ionotropic
file_to_save = results_folder / "Iono_I.svg"

plot_save_scatter_plot(PC1, 
                       Iono_I, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Infragranular Layers - Ionotropic", 
                       x_label="PC1", 
                       y_label="Receptor Density"
                       )

# Infragranular layers - Metabotropic
file_to_save = results_folder / "Metabo_I.svg"

plot_save_scatter_plot(PC1, 
                       Metabo_I, 
                       dataPoints_names=RegionNames_Reduced, 
                       path_name_saved_file=file_to_save, 
                       title="Infragranular Layers - Metabotropic", 
                       x_label="PC1", 
                       y_label="Receptor Density"
                       )

# Examine the relation with the histological gradient of BigBrain
# First align the BigBrain G1 wit hthe receptor data by removing the agranular
# that do not have complete measurements
G1_BigBrain_reduced = G1_BigBrain[indexes_notzeros]

indexes_not_nan = np.argwhere(np.isfinite(G1_BigBrain_reduced))
indexes_not_nan = indexes_not_nan[:,0]

RegionNames_Reduced_Further = RegionNames_Reduced[indexes_not_nan]

#USe this index to reassemble the necessary data is new variables
G1_BigBrain_reduced = G1_BigBrain_reduced[indexes_not_nan]
PC1_reduced = PC1[indexes_not_nan]


file_to_save = results_folder / "PC1_G1BigBrain.svg"

plot_save_scatter_plot(PC1_reduced, 
                       G1_BigBrain_reduced, 
                       dataPoints_names=RegionNames_Reduced_Further, 
                       path_name_saved_file=file_to_save, 
                       title="Natural axis of recepto- and cytoarchitecture", 
                       x_label="Receptoarchitectonic gradient (PC1)", 
                       y_label="Cytoarchitectonic gradient"
                       )

# Examine how the receptor properties, that is, ExcInh, Entropy and overall
# iono/metabotropic densities. PC1 of receptors correlates with the 
# "Cytoarchitectre G1" so overall a similar picture will hold as for the 
# "within receptoarchitecture" analysis

#Entropy across the cytoarchitectonic gradient 
H_I = H_I[indexes_not_nan]
H_G = H_G[indexes_not_nan]
H_S = H_S[indexes_not_nan]

# Infragranular layers
file_to_save = results_folder / "H_I_G1BigBrain.svg"

plot_save_scatter_plot(G1_BigBrain_reduced, 
                       H_I, 
                       dataPoints_names=RegionNames_Reduced_Further, 
                       path_name_saved_file=file_to_save, 
                       title="Infragranular Layers", 
                       x_label="Cytoarchitectonic gradient", 
                       y_label="Entropy of receptor density"
                    )

# Granular layers
file_to_save = results_folder / "H_G_G1BigBrain.svg"

plot_save_scatter_plot(G1_BigBrain_reduced, 
                       H_G, 
                       dataPoints_names=RegionNames_Reduced_Further, 
                       path_name_saved_file=file_to_save, 
                       title="Granular Layers", 
                       x_label="Cytoarchitectonic gradient", 
                       y_label="Entropy of receptor density"
                       )

# Supragranular layers

file_to_save = results_folder / "H_S_G1BigBrain.svg"

plot_save_scatter_plot(G1_BigBrain_reduced, 
                       H_S, 
                       dataPoints_names=RegionNames_Reduced_Further, 
                       path_name_saved_file=file_to_save, 
                       title="Supragranular Layers", 
                       x_label="Cytoarchitectonic gradient", 
                       y_label="Entropy of receptor density"
                       )
    
#Fit an ANCOVA model to uncover if the relation of PC1 and Entropy of receptor
#density is meadiated by the Layer type
H = np.concatenate((H_I, H_G, H_S), 
                   axis=0)
H_ranked = rankdata(H)

#Make a categorical predictor indicating what is supra=1 granular=2 infra=3
supra_index = np.asarray([1]*H_I.size)
granular_index = np.asarray([2]*H_I.size)
infra_index = np.asarray([3]*H_I.size)

Layer = np.concatenate((infra_index, granular_index, supra_index), 
                              axis=0)

G1_BigBrain_concatanated = np.concatenate((G1_BigBrain_reduced, G1_BigBrain_reduced, G1_BigBrain_reduced), 
                                          axis=0)

file_to_save = results_folder / "summary_fit_H_G1BigBrain.txt"

fit_H_G1BigBrain = run_ancova(G1_BigBrain_concatanated, 
                             H_ranked, 
                             Layer, 
                             filename_results=file_to_save)

# Find the receptors that are the the most predictive of the cytoarchitectonic 
# gradient

# Align receptors with the cyto gradient values by removing entries with no 
# values
ReceptorProfiles = X[indexes_not_nan,:]

# Run the rfe with the correct target variable
(MSE_of_rfe_step, 
 FeatureNames_RFE_steps, 
 FeatureScores_RFE_steps, 
 Mean_AllPredictions) = custom_RFE(ReceptorProfiles, 
                                   G1_BigBrain_reduced, 
                                   test_size_perc=0.2, 
                                   iterations=100, 
                                   feature_names=ReceptorNames_I_G_S
                                   )


# Run the rfe with permuted target variable
G1_BigBrain_reduced_null = G1_BigBrain_reduced[np.random.permutation(len(G1_BigBrain_reduced))]

(MSE_of_rfe_step_null, 
 FeatureNames_RFE_steps_null, 
 FeatureScores_RFE_steps_null, 
 Mean_AllPredictions_null) = custom_RFE(ReceptorProfiles, 
                                        G1_BigBrain_reduced_null, 
                                        test_size_perc=0.2, 
                                        iterations=100, 
                                        feature_names=ReceptorNames_I_G_S
                                        )

# Plot results of rfe
mean_MSE = np.mean(MSE_of_rfe_step, axis=0)
mean_MSE_null = np.mean(MSE_of_rfe_step_null, axis=0)

std_MSE = np.std(MSE_of_rfe_step, axis=0)
std_MSE_null = np.std(MSE_of_rfe_step_null, axis=0)

# Auxiliary varible for plotting
size_RFE = MSE_of_rfe_step.shape

#a = [i for i in range(size_RFE[1])]

fig = plt.figure()
fig.set_size_inches(10, 10)  

plt.errorbar(range(size_RFE[1]), mean_MSE, yerr=std_MSE)
plt.errorbar(range(size_RFE[1]), mean_MSE_null, yerr=std_MSE_null)

# Save rfe results figure
file_to_save = results_folder / "rfe_performance.svg"
plt.savefig(file_to_save, format="svg")

# Compute an "importance score" for each receptor. The score is simply the 
# number of iterations that each feature survives all the way up to
# the iteration that was deemed the best
best_iteration = np.where(mean_MSE == mean_MSE.min())
best_iteration = best_iteration[0]

Feature_Scores = np.array([0]*len(ReceptorNames_I_G_S))

for index,feature in enumerate(ReceptorNames_I_G_S):
    present = 0
    for iterations in range(best_iteration[0]):
        features_of_iteration = FeatureNames_RFE_steps[iterations]
        if(features_of_iteration.count(feature) > 0):
            present = present + 1
            
    Feature_Scores[index] = present            
    
# Visualize the scores by stacking them in a "layer-wise" fashion
layer_wise_featurescores = np.vstack((Feature_Scores[30:45:1], 
                            Feature_Scores[15:30:1], 
                            Feature_Scores[0:15:1]))

fig = plt.figure()
fig.set_size_inches(10, 5) 
plt.imshow(layer_wise_featurescores, extent=[0, 
                                             layer_wise_featurescores.shape[1], 
                                             0, 
                                             layer_wise_featurescores.shape[0]])
plt.colorbar()
ax = plt.gca() 
ax.set_xticklabels(list(ReceptorNames))
ax.set_yticklabels(['IG', 'G', 'SG'])
plt.xticks(rotation=90)

plt.ylabel('Layers')
plt.xlabel('Receptors')
plt.title('Consistency of feature selection')

file_to_save = results_folder / 'layer_wise_featurescores.svg'
plt.savefig(file_to_save, format='svg')
 