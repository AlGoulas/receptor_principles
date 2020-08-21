#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import seaborn as sns

# Biplot
def mybiplot(score, coeff, 
             path_name_saved_file=None, 
             score_labels=None, 
             coeff_labels=None):
    '''
    Plot the scores and coefficients derived from a PCA
    
    Input
    -----
    score: ndarray of shape (N,2) of float, where N are the observations and 
        2 are the PC components, usually the 1st and 2nd PCs.
    
    coeff: ndarray of shape (M,2) of float, where M are the nr of features and 
        2 are the coefficients corresponding to the 2 PCs in score.
        
    path_name_saved_file: pathlib.PosixPath object denoting the path and 
        filename of the visualized biplot to be stored.   
        
    score_labels: ndarray of str or list of str of N str corresponding
        to the N observations.
        
    score_labels: ndarray of str or list of str of M str corresponding
        to the M number of features.    
    
    '''
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
    
    # Make color bar of scale PC1 (xs) - this is only for "offline" visualization
    fig = plt.figure()
    plt.colorbar(plt.imshow(np.vstack((xs, xs)), extent=[0,5,0,5]))
    plt.savefig(Path(os.path.dirname(path_name_saved_file)) / "colorbarPC1.svg", 
                format="svg"
                )

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
    '''
    Plot and save a scatterplot of x plotted against y 
    
    Input
    -----
    x: ndarray of shape (N,), int or float, corresponding to x axis values
    
    y: ndarray of shape (N,), int or float, corresponding to y axis values 
    
    path_name_saved_file: pathlib.PosixPath object denoting the path and 
        filename of the visualized scatterplot to be stored. 
        
    title: str, title for the scatterplot 
    
    x_label: str, label for the x axis
    
    y_label: str, label for the y axis   

    '''
    
    data_to_plot = {'PC1':x, 'ReceptorDensity':y}
    df = pd.DataFrame(data_to_plot)

    fig = plt.figure()
    fig.set_size_inches(10, 10)  
      
    sns.regplot(x="PC1", y="ReceptorDensity", data=df,
            scatter_kws={"s":150,
                         "edgecolors":"none"}
            )

    for index, value in enumerate(y):     
        plt.text(x[index], value, dataPoints_names[index], fontsize=30)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(path_name_saved_file, format="svg")
       
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
    '''
    Plot and save a scatterplot of x plotted against y 
    
    Input
    -----
    values: ndarray of shape (N,), int or float, to rank order and plot
    
    labels: ndarray of shape (N,) of str or list of len N,
        where N is the number of values, with the names of each N values 
    
    path_name_saved_file: pathlib.PosixPath object denoting the path and 
        filename of the visualized scatterplot to be stored. 
        
    title: str, title for the scatterplot 
    
    x_label: str, label for the x axis
    
    y_label: str, label for the y axis 
    
    y_min: float, default None, min value to be visualized for the y-axis   
    
    y_max: float, default None, max value to be visualized for the y-axis   

    '''
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
    
def read_area_colors(path_to_file):
    '''
    Read a txt file that stores names and RGB colors 
    
    Input
    -----
    path_to_file: pathlib.PosixPath object denoting the path and 
        filename of the file to be read.
        Each line in the txt file must have the form:
        name;R,G,B where name is a discription of the nature of the 
        entity described, and R G B are the RGB values [0 255]
        
    Output
    ------
    area_colors: dict of len N, where N are the number of lines in the
        path_to_file txt file.
        The dictionary has as keys the names, str, in the txt file and as
        values, str, the RGB values converted to HEX.
    
    '''
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

def plot_colorbars(values, path_name=None, filename=None):
    '''
    Function for plotting a colorbar corresponding to the values
    in values. Values will be plotted as an img as np.vstack((values, values)
    and stored as an svg file.
    Used only for storing colorbars in an svg file for further visualization
    use. 
    
    Input                
    -----
    
    values: ndarray, float or int, with the values for which a colorbar is 
        needed.
        
    path_to_file: pathlib.PosixPath object denoting the path where the svg file
        will be stored
     
    filename: str, the name of the svg file to be stored    
                                            
    '''

    plt.figure()
    plt.colorbar(plt.imshow(np.vstack((values, values)), extent=[0,5,0,5]))
    plt.savefig(path_name / filename, 
                format="svg"
                )
    
def read_replace_write(path_to_file, 
                       path_to_new=None, 
                       filename_new=None,
                       list_to_replace=None,
                       list_new=None
                       ):
    '''
    Read file and replace str i in list_to_replace[i] with list_new[i]
    
    Input
    -----
    path_to_file: pathlib.PosixPath object denoting the full path of the 
        the file to be read.
        
    path_to_new: pathlib.PosixPath object denoting the folder path
       where the file will be stored. 
       
    filename_new: str, denoting the name of file to be stored.
     
    list_to_replace: list of str, containing all the str to be replaced in the
        file with path_to_file full path, i.e.
        list_to_replace[i] will be replaced by list_new[i]  
     
    list_new: list of str, containing all the str for replacing the str in the
        file with path_to_file full path, i.e.
        list_to_replace[i] will be replaced by list_new[i]      
    
    '''
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