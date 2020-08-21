#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from random import randint
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols

# Compute excitation/inhibition ratio
# ExcInh_Index: 1=Excitatory receptor 2=Inhibitory receptor
def compute_excinh(Data_I, Data_G, Data_S, ExcInh_Index):
    '''
    Compute the exc/inh ratio for I, G and S layers
    
    Data_I: ndarray, float, of shape (N,M) where N are the number of datapoints
        and M the number of features (receptor densities) for the I
        (infragranular layers)
        
    Data_G: ndarray, float, of shape (N,M) where N are the number of datapoints
        and M the number of features (receptor densities) for the G
        (granular layers)
        
    Data_S: ndarray, float, of shape (N,M) where N are the number of datapoints
        and M the number of features (receptor densities) for the S
        (supragranular layers)  
        
    ExcInh_Index: ndarray of integers of shape (M,) where M  M the number of 
        features (receptor densities) 1=excitatory 2=inhibitory 
        
    Output
    ------
    
    ExcInh_I: ndarray, float, of of shape (N,) with exc/inh ratio for 
        infragranular layers  
        
    ExcInh_G:  ndarray, float, of of shape (N,) with exc/inh ratio for 
        infragranular layers 
        
    ExcInh_S:  ndarray, float, of of shape (N,) with exc/inh ratio for 
        infragranular layers
        
    indexes_notzeros: ndarray, with the idx of all entries that are non 0
        in ExcInh_I ExcInh_G ExcInh_S      
    
    '''    
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

# Compute density of ionotropic and metabotropic receptors
# ReceptorTypes_IonoMetabo: 1=Ionotropic receptor 2=Metabotropic receptor
def compute_iono_metabo(Data_I, Data_G, Data_S, ReceptorTypes_IonoMetabo):
    '''
    Compute theinotropic and metabotropic receptor density for I, G and S 
    layers
    
    Input
    -----
    
    Data_I: ndarray, float, of shape (N,M) where N are the number of datapoints
        and M the number of features (receptor densities) for the I
        (infragranular layers)
        
    Data_G: ndarray, float, of shape (N,M) where N are the number of datapoints
        and M the number of features (receptor densities) for the G
        (granular layers)
        
    Data_S: ndarray, float, of shape (N,M) where N are the number of datapoints
        and M the number of features (receptor densities) for the S
        (supragranular layers)  
        
    ReceptorTypes_IonoMetabo: ndarray of integers of shape (M,) where M
        the number of features (receptor densities) 1=ionotropic 2=metabotropic 
        
    Output
    ------
    
    Iono_I: ndarray, float, of of shape (N,) with ionotropic densitities for 
        infragranular layers  
        
    Iono_G:  ndarray, float, of of shape (N,) with ionotropic densitities for 
        infragranular layers  
        
    Iono_S:  ndarray, float, of of shape (N,) with ionotropic densitities for 
        infragranular layers  
    
    Metabo_I: ndarray, float, of of shape (N,) with metabotropic densitities 
        for infragranular layers  
        
    Metabo_G:  ndarray, float, of of shape (N,) with metabotropic densitities 
        for infragranular layers  
        
    Metabo_S:  ndarray, float, of of shape (N,) with metabotropic densitities 
        for infragranular layers  
        
    indexes_notzeros: ndarray, with the idx of all entries that are non 0
        in Iono_I Iono_G Iono_S      
    
    ''' 
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

#Calculate entropy of each area across receptors
def calculate_entropy(ReceptData, normalize=True):
    '''
    Calculate the Shannon entropy for the ReceptData observation i based on 
    features M 
    
    Input
    -----
    ReceptData: ndarray, float, of shape (N,M) with N the observations and M
        features.
   
    normalize: bool, default True, if true then normalizes each column of 
        ReceptData by dividing with the max of each column     
     
    Output
    ------
    H: ndarray, float, of shape (N,) denotign the entropy of each of the N 
        observations 
    
    '''
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

def run_ancova(Y, X, Covariate, filename_results=None):
    '''
    Run an ANCOVA analysis with Y as the dependent variable X the predictors 
    and Covariate as the covariate  
    
    Input
    -----
    Y: ndrarray, float, with shape (N,) - the dependent variable
    
    X: ndarray, float, with shape (N,M) - the predictors
    
    Covariate: ndarray, int, of shape (N,) - the covariate that must be have a 
        categorical character 

    filename_results: pathlib.PosixPath object denoting the full path of the
        file to store the results
        
    Output
    ------
    fit: statsmodels.regression.linear_model.RegressionResultsWrapper object    
    '''
    data = {'Y':Y, 'X':X, 'Cov':Covariate}
    df = pd.DataFrame(data)

    formula = 'Y ~ X * C(Cov)'  # ANCOVA formula

    #Fit the model
    lm = ols(formula, df)
    fit = lm.fit()

    #Save the results
    print(fit.summary(), file=open(filename_results, "w")) 
    
    return fit 

def custom_RFE(X, Y, test_size_perc=0.2, iterations=100, feature_names=None):
    '''
    Perform a recursive feature elimination with svr. This implementation
    does not arrive a final rediced feature set, but removes 1 feature at a 
    time recursively. Predictions are based on monte carlo cv.
    
    Input
    -----
    X: ndarray, float, of shape (N,M) where N the observatiosn and M the 
        features
    
    Y: ndarray, float, of shape (N,) - the variable to be predicted
    
    test_size_perc: float (0 1], default 0.2, specifying the percentage of 
        X to be used as a test set.

    iterations: int, default 100, the iterations for the monte carlo cv  

    feature_names: list of str with len M with names for each of the 
        M features    

    Output
    ------
    
    MSE_of_rfe_step: ndarray, float, of shape (iterations, K) with K the total
        number of recursive feature steps 
    
    FeatureNames_RFE_steps: list of lists of str, with len K. Each list of str
        FeatureNames_RFE_steps[i] contains the feature_names of the features 
        that were selected in the recursive feature step i, with i=1,2...K  
    
    FeatureScores_RFE_steps: list of ndarrays, with len K. Each ndarray
        FeatureScores_RFE_steps[i] contains the number of times that the 
        features in FeatureNames_RFE_steps[i] were selected across iterations
        times (the monte carlo cv iterations). Thus, if iterations=100, then
        e.g., 80 means that this feature was selected 80/100 times as the best
        feature in the kth recursive feature elimination step.
    
    Mean_AllPredictions: ndarray, float, of shape (N,K)
        each column K contains the mean predictions for Y at recursive feature
        step K. In total, we have as many steps K as features M.                     
    
    '''
    svr = SVR(kernel="linear", C=1.0)
    # Initialize variables to keep the relevant info
    size_X = X.shape
    total_rfe_steps = size_X[1]-1
    MSE_of_rfe_step = np.array([[0.]*total_rfe_steps]*iterations)
    
    FeatureNames_RFE_steps=[]
    FeatureScores_RFE_steps=[]

    # This object will help us keep track of the positions by offering the 
    # indexes of the train and test set    
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
                  
            # Mark the features that are selected in this iteration 
            # (marked with ones)
            feature_selected = rfe.ranking_
            feature_selected[feature_selected > 1] = 0
            
            # This vector contains the features that are ranked the highest
            # Hence the higher these values the most consistent the feature 
            # selection
            all_feature_selected = all_feature_selected + feature_selected 
            
        # Store the mean mse across all iternal iterations for this rfe step            
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
        X = X[:, indexes_shrink]
        
        # Shrink the names of the features
        res_list = [feature_names[i] for i in indexes_shrink]
        FeatureNames_RFE_steps.append(res_list)
        
        FeatureScores_RFE_steps.append(all_feature_selected[indexes_shrink])

        #Update here the feature names list
        feature_names = res_list[:]

    return MSE_of_rfe_step, FeatureNames_RFE_steps, FeatureScores_RFE_steps, Mean_AllPredictions 