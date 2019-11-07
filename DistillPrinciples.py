#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from random import randint

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr
from scipy.stats import rankdata
#from scipy.stats import zscore

from statsmodels.formula.api import ols

#import statsmodels.api as sm

import seaborn as sns

import pandas as pd

#For rfe and svr
from sklearn.svm import SVR
#from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import ShuffleSplit

#sns.set(context='poster', style='white')

#aAssume fonts are installed when exporting in .svg
plt.rcParams['svg.fonttype'] = 'none'

#Compute excitation/inhibition ratio
#ExcInh_Index: 1=Excitatory receptor 2=Inhibitory receptor
def ComputeExcInh(Data_I, Data_G, Data_S, ExcInh_Index):
    
    #Compute the sum of excitatory and inhitory receptors
    result = np.where(ExcInh_Index == 1)
    indexes_exc = result[0]
    
    result = np.where(ExcInh_Index == 2)
    indexes_inh = result[0]
    
    Exc_I = np.mean(Data_I[:,indexes_exc],1)
    Exc_G = np.mean(Data_G[:,indexes_exc],1)
    Exc_S = np.mean(Data_S[:,indexes_exc],1)
    
    #Granular dataset contains 0s for agranular areas - reassemble data
    #without this area
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


#Compute excitation/inhibition ratio
#ReceptorTypes_IonoMetabo: 1=Ionotropic receptor 2=Metabotropic receptor
def ComputeIonoMetabo(Data_I, Data_G, Data_S, ReceptorTypes_IonoMetabo):
    
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



#Biplot
def mybiplot(score, coeff, path_name_saved_file, score_labels=None, coeff_labels=None):
    
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
                     color = 'g', ha = 'center', va = 'center', fontsize=30)
            
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    #plt.xlabel("PC{}".format(1))
    #plt.ylabel("PC{}".format(2))
    plt.grid()
    #plt.show()
    
    plt.savefig(path_name_saved_file, format="svg")



#Calculate entropy of each area across receptors
def CalculateEntropy(ReceptData, normalize=True):
    
    #If normalization is requested then normalize each column of 
    #ReceptData by dividing with the max of each column. 
    #The ranges of receptor values vary so entropy could get 
    #influenced by such range differences
    
    #Get max of each column and divide each one by this max
    if(normalize==True): 
        
        max_column = np.max(ReceptData,0)
        ReceptData = ReceptData / max_column
  
    #Calculate the entropy for each area
    data_size=np.shape(ReceptData)
    H = np.zeros(data_size[0])
    
    for area in range(data_size[0]):
        
        current_receptor_profile = ReceptData[area,:]
        current_receptor_profile = np.divide(current_receptor_profile,
                                             np.sum(current_receptor_profile))
        
        H[area] = -np.sum(current_receptor_profile * np.log(current_receptor_profile)) \
        / np.log(len(current_receptor_profile));
    
    return H



#Plot and save scatterplot
def PlotSaveScatterPlot(x, y, dataPoints_names, path_name_saved_file, title, x_label, y_label): 
    
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
    
    
def RunAncova(Y, X, Covariate, filename_results):
   
    data = {'Y':Y, 'X':X, 'Cov':Covariate}
    df = pd.DataFrame(data)

    formula = 'Y ~ X * C(Cov)'  # ANCOVA formula

    #Fit the model
    lm = ols(formula, df)
    fit = lm.fit()

    #Save the results
    print(fit.summary(), file=open(filename_results, "w")) 
    
    return fit 

    
def CustomRFE(X, Y, test_size_perc, iterations, feature_names):
    
    svr = SVR(kernel="linear", C=1.0)
    
    #This is not necessary and probably a bias! Check it out
    #Y = zscore(Y)
    
    #Initialize variables to keep the relevant info
    size_X = X.shape
    total_rfe_steps = size_X[1]-1
    #MSE_of_rfe_step = np.array(np.array([0.]*total_rfe_steps))
    MSE_of_rfe_step = np.array([[0.]*total_rfe_steps]*iterations)
    
    FeatureNames_RFE_steps=[]
    FeatureScores_RFE_steps=[]
    
    #feature_index = [i for i in range(size_X[1])]
    #This object will help us keep track of the positions by offering the 
    #indexes of the train and test set
    #sp = ShuffleSplit(n_splits=1, test_size=test_size_perc)
    
    size_Y = Y.shape
    
    Mean_AllPredictions = np.asarray([[0.] * total_rfe_steps] * size_Y[0])
    #AllActual = np.asarray([[0.] * iterations] * size_Y[0])
    
    for rfe_steps in range(0, total_rfe_steps):

        size_X = X.shape
        features = size_X[1] - 1
        all_feature_selected = np.array([0]*size_X[1])
        
        all_mse = np.array([0.]*iterations)
        
        rfe = RFE(estimator=svr, n_features_to_select=features, step=1, verbose=0)
        AllPredictions = np.asarray([[0.] * iterations] * size_Y[0])
        
        for iter in range(0, iterations):
        
            #rfe = RFE(estimator=svr, n_features_to_select=features, step=1, verbose=0)
    
            #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_perc)

            index = np.random.permutation(size_Y[0])
            
            split_position = np.round(test_size_perc * size_Y[0])
            
            train_index = index[int(split_position):]  
            test_index = index[:int(split_position)]  

            X_train, X_test = X[train_index,:], X[test_index,:] 
            Y_train, Y_test = Y[train_index], Y[test_index]
    
#            X_train_zscored = scaler.fit_transform(X_train)
#            X_test_zscored = scaler.fit_transform(X_test)
#    
#            Y_train_zscored = zscore(Y_train)
#            Y_test_zscored = zscore(Y_test)
            
            mean_X_train = np.mean(X_train, axis=0)
            std_X_train = np.std(X_train, axis=0)
            
            X_train_zscored = (X_train - mean_X_train) / std_X_train
            
            #use the mean and std of the training set to scale the test set
            X_test_zscored = (X_test - mean_X_train) / std_X_train
            
            #Compute the mean and std for the target variable 
            #normalize train and test Y
            mean_Y_train = np.mean(Y_train)
            std_Y_train = np.std(Y_train)
    
            Y_train_zscored = (Y_train - mean_Y_train) / std_Y_train
            Y_test_zscored = (Y_test - mean_Y_train) / std_Y_train
            
            #Fit the model and predict from the test set
            rfe.fit(X_train_zscored, Y_train_zscored)
            Y_predicted = rfe.predict(X_test_zscored)
            
            #Store the mse
            all_mse[iter] = mean_squared_error(Y_test_zscored, Y_predicted)
            
            AllPredictions[test_index, iter] = Y_predicted
            #AllActual[test_index, iter] = Y_test
                  
            #Mark the features that are selcted in this iteration 
            #(marked with ones)
            feature_selected = rfe.ranking_
            feature_selected[feature_selected > 1] = 0
            
            #This vector contains the features that are ranked the highest
            #Hence the higher these values the most consistent the feature 
            #selection
            
            all_feature_selected = all_feature_selected + feature_selected 
            
            
        #Store the mean mse across all iternal iterations for this rfe step    
        #MSE_of_rfe_step[rfe_steps] = np.mean(all_mse)
        
        MSE_of_rfe_step[:, rfe_steps] = all_mse
        
        #Here we have to summarize all the predictions by averaging the 
        #predictions across the terations BUT ONLY by taking into account the 
        #times that each datapoint of the target variable was taken into 
        #account
        
        sum_AllPredictions = np.sum(AllPredictions, axis=1)
        ind = np.where(AllPredictions != 0)
        
        AllPredictions[ind] = 1
        timesiterated_AllPredictions = np.sum(AllPredictions, axis=1)
        
        Mean_AllPredictions[:, rfe_steps] = sum_AllPredictions / timesiterated_AllPredictions
        
        #Reduce the feature matrix by discarding the feature with the smallest 
        #consistency score
        indexes_min = np.where(all_feature_selected == all_feature_selected.min())
        indexes_min = indexes_min[0]
        
        #In case of more than one feature with minimum value select
        #one of these features at random
        if(len(indexes_min) > 1):
           indexes_min = indexes_min[randint(0, len(indexes_min)-1)]
        
        all_feature_selected[indexes_min] = -1
        indexes_shrink = np.where(all_feature_selected > -1)
        indexes_shrink = indexes_shrink[0]
        
        #Shrink the feature matrix by excluding the feature with the lowest 
        #score
        X = X[:,indexes_shrink]
        
        #Shrink the names of the features
        #FeatureNames_RFE_steps.append(feature_names[indexes_shrink])
        
        res_list = [feature_names[i] for i in indexes_shrink]
        
        FeatureNames_RFE_steps.append(res_list)
        FeatureScores_RFE_steps.append(all_feature_selected[indexes_shrink])

        #Update here the feature names list
        feature_names = res_list[:]

    return MSE_of_rfe_step, FeatureNames_RFE_steps, FeatureScores_RFE_steps, Mean_AllPredictions 

    
    
#Analyze the data     

#Calculate excitation inhibition 
ExcInh_I, ExcInh_G, ExcInh_S, indexes_notzeros = ComputeExcInh(ReceptData_I, 
                                                               ReceptData_G, 
                                                               ReceptData_S, 
                                                               ReceptorTypes_ExcInh)

#Keep only the data with values - hence remove agranular areas
ReceptData_Reduced_I = ReceptData_I[indexes_notzeros, :]
ReceptData_Reduced_G = ReceptData_G[indexes_notzeros, :]
ReceptData_Reduced_S = ReceptData_S[indexes_notzeros, :]

ExcInh_I = ExcInh_I[indexes_notzeros]
ExcInh_S = ExcInh_S[indexes_notzeros]

RegionNames_Reduced = np.array(RegionNames)[indexes_notzeros]

#Calculate Entropy

#Normalize densities per receptor

max_ReceptData_Reduced_I = np.max(ReceptData_Reduced_I, axis=0)
max_ReceptData_Reduced_G = np.max(ReceptData_Reduced_G, axis=0)
max_ReceptData_Reduced_S = np.max(ReceptData_Reduced_S, axis=0)

ReceptData_Reduced_I_norm = ReceptData_Reduced_I / max_ReceptData_Reduced_I
ReceptData_Reduced_G_norm = ReceptData_Reduced_G / max_ReceptData_Reduced_G
ReceptData_Reduced_S_norm = ReceptData_Reduced_S / max_ReceptData_Reduced_S

H_I = CalculateEntropy(ReceptData_Reduced_I_norm)
H_G = CalculateEntropy(ReceptData_Reduced_G_norm)
H_S = CalculateEntropy(ReceptData_Reduced_S_norm)

#H_I = CalculateEntropy(ReceptData_Reduced_I)
#H_G = CalculateEntropy(ReceptData_Reduced_G)
#H_S = CalculateEntropy(ReceptData_Reduced_S)

#Create a list of receptor names with a prefix indicatign the layer 
#(for PCA visualization)
ReceptorNames_I_G_S = [ ]

for i in range(len(ReceptorNames)*3): 
    
    if i >= 0 and i <= 14:
        ReceptorNames_I_G_S.append(ReceptorNames[i] + '_I')
    
    if i > 14 and i <= 29:
        ReceptorNames_I_G_S.append(ReceptorNames[i-15] + '_G')
    
    if i > 29 and i <= 44:
        ReceptorNames_I_G_S.append(ReceptorNames[i-30] + '_S')

#Run PCA on the receptor data after they are z-scored
X = np.concatenate((ReceptData_Reduced_I, ReceptData_Reduced_G, ReceptData_Reduced_S), axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
scores = pca.fit_transform(X_scaled)
coeff = np.transpose(pca.components_[0:2, :])# Do we have to transpose? Yes!

PC1 = scores[:,0] 
PC1_ranked = rankdata(PC1) 

#Visualize with a biplot - this visualization can be ameliorated - TODO
#mybiplot(scores[:, 0:2], coeff, "results/biplot.svg", 
#         RegionNames_Reduced, 
#         ReceptorNames_I_G_S)


PC1_2 = scores[:,0:2]
PC1_2[:,1] = -1*PC1_2[:,1]

coeff[:,1] = -1*coeff[:,1] 

mybiplot(PC1_2, coeff, "results/biplot.svg", 
         RegionNames_Reduced, 
         ReceptorNames_I_G_S)

#Calcualte the correlation between EscInh across PC1 and plot the data
rho_ExcInh_S, pval_ExcInh_S = spearmanr(PC1, ExcInh_S)
rho_ExcInh_G, pval_ExcInh_G = spearmanr(PC1, ExcInh_G)  
rho_ExcInh_I, pval_ExcInh_I = spearmanr(PC1, ExcInh_I)
    
# Plot the relation between Exc/Inh and PC1

#Infragranular layers
PlotSaveScatterPlot(PC1, ExcInh_I, RegionNames_Reduced, 
                    "results/ExcInh_I.svg", "Infragranular Layers", "PC1", "Exc/Inh receptor density")

#Granular layers
PlotSaveScatterPlot(PC1, ExcInh_G, RegionNames_Reduced, 
                    "results/ExcInh_G.svg", "Granular Layers", "PC1", "Exc/Inh receptor density")

#Supragranular layers
PlotSaveScatterPlot(PC1, ExcInh_S, RegionNames_Reduced, 
                    "results/ExcInh_S.svg", "Supragranular Layers", "PC1", "Exc/Inh receptor density")

#Run an ANCOVA model to test if the Exc/Inh and PC1 slopes are layer specific
ConcPC1_ranked = np.concatenate((PC1_ranked, PC1_ranked, PC1_ranked), axis=0)


#ExcInh = np.concatenate((ExcInh_I_ranked, ExcInh_G_ranked, ExcInh_S_ranked), axis=0)

ExcInh = np.concatenate((ExcInh_I, ExcInh_G, ExcInh_S), axis=0)
ExcInh_ranked = rankdata(ExcInh)

#Make a categorical predictor indicating what is supra=1 granular=2 infra=3

supra_index = np.asarray([1]*ExcInh_I.size)
granular_index = np.asarray([2]*ExcInh_I.size)
infra_index = np.asarray([3]*ExcInh_I.size)

Layer = np.concatenate((infra_index, granular_index, supra_index), 
                              axis=0)

fit_ExcInh = RunAncova(ConcPC1_ranked, ExcInh_ranked, Layer, 
                       "results/summary_fit_ExcInh.txt")


#Plot and save a boxplot for a summary of overall Exc/Inh in each layer
#Use unranked data for better interpertabilty
data_ExcInh_LayerWise = {'PC1':ConcPC1_ranked, 'ExcInh':ExcInh, 'Layer':Layer}
df = pd.DataFrame(data_ExcInh_LayerWise)

fig = plt.figure()
fig.set_size_inches(10, 10)  
sns.boxplot(x="Layer", y="ExcInh", data=df, palette="Set3")

plt.savefig("results/ExcInh_LayerWise.svg", format="svg")


#Compute the correaltion between entropy of receptor density and PC1
rho_H_S, pval_H_S = spearmanr(PC1, H_S)
rho_H_I, pval_H_I = spearmanr(PC1, H_I)
rho_H_G, pval_H_G = spearmanr(PC1, H_G)

#Plot the correlation between Entropy across PC1 

#Infragranular layers
PlotSaveScatterPlot(PC1, H_I, RegionNames_Reduced, 
                    "results/H_I.svg", "Infragranular Layers", "PC1", 
                    "Entropy of receptor density")

#Granular layers
PlotSaveScatterPlot(PC1, H_G, RegionNames_Reduced, 
                    "results/H_G.svg", "Granular Layers", "PC1", 
                    "Entropy of receptor density")

#Supragranular layers
PlotSaveScatterPlot(PC1, H_S, RegionNames_Reduced, 
                    "results/H_S.svg", "Supragranular Layers", "PC1", 
                    "Entropy of receptor density")
    

#Fit an ANCOVA model to uncover if the relation of PC1 and Entropy of receptor
#density is meadiated by the Layer type

H = np.concatenate((H_I, H_G, H_S), axis=0)
H_ranked = rankdata(H)

#Make a categorical predictor indicating what is supra=1 granular=2 infra=3

supra_index = np.asarray([1]*H_I.size)
granular_index = np.asarray([2]*H_I.size)
infra_index = np.asarray([3]*H_I.size)

Layer = np.concatenate((infra_index, granular_index, supra_index), 
                              axis=0)


fit_H = RunAncova(ConcPC1_ranked, H_ranked, Layer, 
                       "results/summary_fit_H.txt")


#Plot and save a boxplot for a summary of overall Entropy in each lamiane
#Use unranked data for better interpertabilty
data_H_LayerWise = {'PC1':ConcPC1_ranked, 'H':H, 'Layer':Layer}
df = pd.DataFrame(data_H_LayerWise)

fig = plt.figure()
fig.set_size_inches(10, 10)  
sns.boxplot(x="Layer", y="H", data=df, palette="Set3")

plt.savefig("results/H_LayerWise.svg", format="svg")


#Estimate overall density of ionotropic and metabotropic receptors and how
#they relate to PC1
Iono_I, Iono_G, Iono_S, Metabo_I, Metabo_G, Metabo_S, indexes_notzeros_ionometabo = ComputeIonoMetabo(ReceptData_I,
                                                                                                     ReceptData_G,
                                                                                                     ReceptData_S,
                                                                                                     ReceptorTypes_IonoMetabo)

Iono_I = Iono_I[indexes_notzeros_ionometabo]
Iono_S = Iono_S[indexes_notzeros_ionometabo]           

Metabo_I = Metabo_I[indexes_notzeros_ionometabo]
Metabo_S = Metabo_S[indexes_notzeros_ionometabo]  

#Make a categorical predictor indicating that iono =1 and metabo =2

ConcPC1_ranked = np.concatenate((PC1_ranked, PC1_ranked), axis=0)

#This categorical predictor will be used for all I, G, S models since
#the size and arrangement of the predictors is identical
index_iono = np.asarray([1]*Iono_I.size)
index_metabo = np.asarray([2]*Metabo_I.size)

ReceptorType = np.concatenate((index_iono, index_metabo), axis=0)

IonoMetaboDensity_I = np.concatenate((Iono_I, Metabo_I), axis=0)
IonoMetaboDensity_I_ranked = rankdata(IonoMetaboDensity_I)

IonoMetaboDensity_G = np.concatenate((Iono_G, Metabo_G), axis=0)
IonoMetaboDensity_G_ranked = rankdata(IonoMetaboDensity_G)

IonoMetaboDensity_S = np.concatenate((Iono_S, Metabo_S), axis=0)
IonoMetaboDensity_S_ranked = rankdata(IonoMetaboDensity_S)

#Fit the ancova models and save the results
fit_IonoMetaboDensity_I = RunAncova(ConcPC1_ranked, IonoMetaboDensity_I_ranked, ReceptorType, 
                       "results/summary_fit_IonoMetabo_I.txt")

fit_IonoMetaboDensity_G = RunAncova(ConcPC1_ranked, IonoMetaboDensity_G_ranked, ReceptorType, 
                       "results/summary_fit_IonoMetabo_G.txt")

fit_IonoMetaboDensity_S = RunAncova(ConcPC1_ranked, IonoMetaboDensity_S_ranked, ReceptorType, 
                       "results/summary_fit_IonoMetabo_S.txt")


# Plot seperately the relation of receptor density and PC1 for iono and metabo
#receptors

#Supragranular layers - Ionotropic
PlotSaveScatterPlot(PC1, Iono_S, RegionNames_Reduced, 
                    "results/Iono_S.svg", "Supragranular Layers - Ionotropic", "PC1", "Receptor Density")

#Supragranular layers - Metabotropic
PlotSaveScatterPlot(PC1, Metabo_S, RegionNames_Reduced, 
                    "results/Metabo_S.svg", "Supragranular Layers - Metabotropic", "PC1", "Receptor Density")

#Granular layers - Ionotropic
PlotSaveScatterPlot(PC1, Iono_G, RegionNames_Reduced, 
                    "results/Iono_G.svg", "Granular Layers - Ionotropic", "PC1", "Receptor Density")

#Granular layers - Metabotropic
PlotSaveScatterPlot(PC1, Metabo_G, RegionNames_Reduced, 
                    "results/Metabo_G.svg", "Granular Layers - Metabotropic", "PC1", "Receptor Density")

#Infragranular layers - Ionotropic
PlotSaveScatterPlot(PC1, Iono_I, RegionNames_Reduced, 
                    "results/Iono_I.svg", "Infragranular Layers - Ionotropic", "PC1", "Receptor Density")

#Infragranular layers - Metabotropic
PlotSaveScatterPlot(PC1, Metabo_I, RegionNames_Reduced, 
                    "results/Metabo_I.svg", "Infragranular Layers - Metabotropic", "PC1", "Receptor Density")


#Examine the relation with the histological gradient of BigBrain
#First align the BigBrain G1 wit hthe receptor data by removing the agranular
#that do not have complete measurements

G1_BigBrain_reduced = G1_BigBrain[indexes_notzeros]

indexes_not_nan = np.argwhere(np.isfinite(G1_BigBrain_reduced))
indexes_not_nan = indexes_not_nan[:,0]

RegionNames_Reduced_Further = RegionNames_Reduced[indexes_not_nan]

#USe this index to reassemble the necessary data is new variables
G1_BigBrain_reduced = G1_BigBrain_reduced[indexes_not_nan]
PC1_reduced = PC1[indexes_not_nan]

PlotSaveScatterPlot(PC1_reduced, G1_BigBrain_reduced, RegionNames_Reduced_Further, 
                    "results/PC1_G1BigBrain.svg", "Natural axis of recepto- and cytoarchitecture", "Receptoarchitectonic gradient (PC1)", "Cytoarchitectonic gradient")


#Examine how the receptor properties, that is, ExcInh, Entropy and overall
#iono/metabotropic densities. PC1 of receptors correlates with the 
#"Cytoarchitectre G1" so overall a similar picture will hold as for the 
#"within receptoarchitecture" analysis

#Entropy across the cytoarchitectonic gradient 

H_I = H_I[indexes_not_nan]
H_G = H_G[indexes_not_nan]
H_S = H_S[indexes_not_nan]

#Infragranular layers
PlotSaveScatterPlot(G1_BigBrain_reduced, H_I, RegionNames_Reduced_Further, 
                    "results/H_I_G1BigBrain.svg", "Infragranular Layers", "Cytoarchitectonic gradient", 
                    "Entropy of receptor density")

#Granular layers
PlotSaveScatterPlot(G1_BigBrain_reduced, H_G, RegionNames_Reduced_Further, 
                    "results/H_G_G1BigBrain.svg", "Granular Layers", "Cytoarchitectonic gradient", 
                    "Entropy of receptor density")

#Supragranular layers
PlotSaveScatterPlot(G1_BigBrain_reduced, H_S, RegionNames_Reduced_Further, 
                    "results/H_S_G1BigBrain.svg", "Supragranular Layers", "Cytoarchitectonic gradient", 
                    "Entropy of receptor density")
    

#Fit an ANCOVA model to uncover if the relation of PC1 and Entropy of receptor
#density is meadiated by the Layer type

H = np.concatenate((H_I, H_G, H_S), axis=0)
H_ranked = rankdata(H)

#Make a categorical predictor indicating what is supra=1 granular=2 infra=3

supra_index = np.asarray([1]*H_I.size)
granular_index = np.asarray([2]*H_I.size)
infra_index = np.asarray([3]*H_I.size)

Layer = np.concatenate((infra_index, granular_index, supra_index), 
                              axis=0)

G1_BigBrain_concatanated = np.concatenate((G1_BigBrain_reduced, G1_BigBrain_reduced, G1_BigBrain_reduced), axis=0)

fit_H_G1BigBrain = RunAncova(G1_BigBrain_concatanated, H_ranked, Layer, 
                       "results/summary_fit_H_G1BigBrain.txt")


#Find the receptors that are the the most predictive of the cytoarchitectonic gradient

#Align receptors with the cyto gradient values by removing entries with no 
#values

ReceptorProfiles = X[indexes_not_nan,:]

#Run the rfe with the correct target variable
MSE_of_rfe_step, FeatureNames_RFE_steps, FeatureScores_RFE_steps, Mean_AllPredictions = CustomRFE(ReceptorProfiles, 
                                                                             G1_BigBrain_reduced, 
                                                                             0.2, 
                                                                             1000, 
                                                                             ReceptorNames_I_G_S)


#Run the rfe with permuted target variable
G1_BigBrain_reduced_null = G1_BigBrain_reduced[np.random.permutation(len(G1_BigBrain_reduced))]

MSE_of_rfe_step_null, FeatureNames_RFE_steps_null, FeatureScores_RFE_steps_null, Mean_AllPredictions_null = CustomRFE(ReceptorProfiles, 
                                                                             G1_BigBrain_reduced_null, 
                                                                             0.2, 
                                                                             1000, 
                                                                             ReceptorNames_I_G_S)

#Plor results of rfe
mean_MSE = np.mean(MSE_of_rfe_step, axis=0)
mean_MSE_null = np.mean(MSE_of_rfe_step_null, axis=0)

std_MSE = np.std(MSE_of_rfe_step, axis=0)
std_MSE_null = np.std(MSE_of_rfe_step_null, axis=0)

#Auxiliary varible for plotting
size_RFE = MSE_of_rfe_step.shape

#a = [i for i in range(size_RFE[1])]

fig = plt.figure()
fig.set_size_inches(10, 10)  

plt.errorbar(range(size_RFE[1]), mean_MSE, yerr=std_MSE)
plt.errorbar(range(size_RFE[1]), mean_MSE_null, yerr=std_MSE_null)

#Compute an "importance score" for each receptor. The score is simply the 
#number of iterations that each feature survives all the way up to
#the iteration that was deemed the best

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
    
#Visualize the scores by stacking them in a "layer-wise" fashion
layer_wise_featurescores = np.vstack((Feature_Scores[30:45:1], 
                            Feature_Scores[15:30:1], 
                            Feature_Scores[0:15:1]))

fig = plt.figure()
fig.set_size_inches(10, 10) 

plt.imshow(layer_wise_featurescores)



# Test plots for average results across layers
#AllReceptData = ReceptData_Reduced_I + ReceptData_Reduced_G + ReceptData_Reduced_S
#AllReceptData = np.round(AllReceptData)
#AllReceptData_scaled = scaler.fit_transform(AllReceptData)
#
#scores = pca.fit_transform(AllReceptData_scaled)
#coeff = np.transpose(pca.components_[0:2, :])# Do we have to transpose? Yes!
#
#PC1_2 = scores[:,0:2]
#PC1_2[:,1] = -1*PC1_2[:,1]
#
#coeff[:,1] = -1*coeff[:,1]
#
#mybiplot(PC1_2, coeff,  "results/biplot_averagedensities.svg",
#         RegionNames_Reduced, 
#         ReceptorNames)
#
#max_AllReceptData = np.max(AllReceptData, axis=0)
#
#AllReceptData_norm = AllReceptData / max_AllReceptData
#
#
#H = CalculateEntropy(AllReceptData_norm)
#
#PlotSaveScatterPlot(PC1_2[:,0], H, RegionNames_Reduced, 
#                    "results/H_PC1_alldensities.svg", "Entropy-PC1", "PC1", 
#                    "H")


#Perform a CDA with lobes as groups. Even though we are interested in the 
#natural axis formed by the receptors densities, it provides a lobe-wise
#summary that may be useful to "macroscopic-based" researchers  

 