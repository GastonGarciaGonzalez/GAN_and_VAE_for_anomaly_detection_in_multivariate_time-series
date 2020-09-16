# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:49:43 2020

@author: usuario
"""
"""
Created on Tue Apr 14 19:15:50 2020

@author: Gastón García González

Important:
    The base structure of this script was extruded from the work done on:
    https://github.com/LiDan456/GAN-AD
    
    CLi, D., Chen, D., Goh, J., & Ng, S. K. (2018).
    Anomaly detection with generative adversarial networks for multivariate time series.
    arXiv preprint arXiv:1809.04758.
    
    This last work is also based on a previous work (If you want to see the source code,
    please refer to):
    https://github.com/ratschlab/RGAN
    
    Esteban, C., Hyland, S. L., & Rätsch, G. (2017). 
    Real-valued (medical) time series generation with recurrent conditional gans. 
    arXiv preprint arXiv:1706.02633.     
"""
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib

def get_data(seq_length=12, 
             shift_sample=1, 
             aggregate_sec=10, 
             scaler=None, 
             scaler_type='standar', 
             path='.', start_time=None, 
             end_time=None, name_scaler='', 
             save_npy=False):
    
    try:
        samples = np.load(path+"samples_"+str(aggregate_sec)+"_"+str(seq_length)+"_"+str(shift_sample)+".npy")
        labels_samples= np.load(path+"labels_"+str(aggregate_sec)+"_"+str(seq_length)+"_"+str(shift_sample)+".npy")
        timestamp = np.load(path+"time_"+str(aggregate_sec)+"_"+str(seq_length)+"_"+str(shift_sample)+".npy")
        columns_names = None
        print('Loaded data from .npy')

    except IOError:
        print('Failed to load the data from .npy, loading from csv')

        file = glob.glob(path)
        
        data = pd.read_csv(file[0], sep=',')
        columns_names = data.columns
        data[columns_names[0]] = pd.to_datetime(data[columns_names[0]], dayfirst=True)
        data = data.set_index(columns_names[0])

        # aggregate 
        length_data = data.shape[0]
        data = data.resample(str(aggregate_sec)+'S').mean()
        # labels from the last column
        labels = data[columns_names[-1]].round().to_numpy()
        data = data.drop(columns_names[-1], axis=1)
        # timestamp from the index
        time = data.index.to_numpy()
        
        #drop by time
        if start_time != None:
            data = data.loc[(start_time <= data.index)]
        if end_time != None:
            data = data.loc[(data.index < end_time)]

        # scale data, mean=0, std=1
        if scaler == None:
            if scaler_type == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standar':
                scaler = StandardScaler()
                
            scaler.fit(data.values)
            scaler_filename = "scaler_"+name_scaler+".save"
            joblib.dump(scaler, scaler_filename) 

        data = scaler.transform(data.values)
      
        samples = []
        labels_samples = []
        timestamp = []

        i = 0
        while (i*shift_sample + seq_length) < (data.shape[0]):
            samples.append(data[i*shift_sample:i*shift_sample + seq_length,:])
            labels_samples.append(labels[i*shift_sample:i*shift_sample + seq_length])
            timestamp.append(time[i*shift_sample:i*shift_sample + seq_length])
            i += 1

        samples = np.array(samples)
        labels_samples = np.array(labels_samples)
        timestamp = np.array(timestamp)
        
        if save_npy:
            np.save(path+"samples_"+str(aggregate_sec)+"_"+str(seq_length)+"_"+str(shift_sample)+".npy", samples)
            np.save(path+"labels_"+str(aggregate_sec)+"_"+str(seq_length)+"_"+str(shift_sample)+".npy", labels_samples)
            np.save(path+"time_"+str(aggregate_sec)+"_"+str(seq_length)+"_"+str(shift_sample)+".npy", timestamp)
            

    return samples, labels_samples, scaler, columns_names, timestamp, length_data