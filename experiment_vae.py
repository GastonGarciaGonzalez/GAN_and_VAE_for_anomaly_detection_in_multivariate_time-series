# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:31:47 2020

@author: usuario
"""

import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from MMD import MMD2
from datetime import datetime, timedelta

import numpy as np
import samples
import json
import time
import random
import pandas as pd
from datetime import datetime
import csv
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model

tf.keras.backend.set_floatx('float64')
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K

from tensorflow.keras import optimizers


#%%
#%%
#date of the experiment
date_exp = str(datetime.now().strftime("%d-%m-%Y %H_%M_%S"))

#settings
settings_path = './settings/' + 'train_vae_swat' + '.txt'
print('Loading settings from', settings_path)
settings = json.load(open(settings_path, 'r'))

# --- get data, split --- #
samp, labels, scaler, columns_names, timestamp, length_data = samples.get_data(settings["seq_length"],
                                                                   settings["seq_step"],
                                                                   settings["aggregate"],
                                                                   scaler=None, path=settings["data_load_from"],
                                                                   scaler_type=settings["scaler_type"],
                                                                   start_time=None,
                                                                   name_scaler=date_exp)
#sigma value to calculate MMD
sigma = np.std(samp)
print('Data train shape:', samp.shape)

#Experiment settings
data_exp_now = {'date': date_exp, 
                'data': settings['data'],
                'path': settings["data_load_from"], 
                'data_size': length_data,
                'num_muestras_train': samp.shape[0], 
                'seq_length': settings['seq_length'], 
                'seq_step': settings["seq_step"], 
                'aggregate': settings["aggregate"], 
                'variables': settings["variables"],
                'hidden_units_e': settings['hidden_units_e'], 
                'hidden_units_d': settings['hidden_units_d'], 
                'latent_dim': settings['latent_dim'], 
                'learning_rate': settings['learning_rate'], 
                'decay_step': settings['decay_step'], 
                'decay_rate': settings['decay_rate'], 
                'batch_size': settings['batch_size'], 
                'num_epochs': settings['num_epochs'], 
                'scaler_type': settings['scaler_type'],
                'path_scaler': "scaler_"+date_exp+".save",
                'scaler': scaler,
                'path_vae_model': ".//models//vae_model_"+date_exp+".h5",
                'training_time': 0
                }

samp = np.reshape(samp, (samp.shape[0], samp.shape[1], samp.shape[2],1))

#sigma value to calculate MMD
sigma = np.std(samp)
print('Data train shape:', samp.shape)
#%%

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#%%
# network parameters
input_shape = (settings['seq_length'], settings['variables'], 1)
intermediate_dim = settings['hidden_units_e']
batch_size = settings['batch_size']
latent_dim = settings['latent_dim']
epochs = settings['num_epochs']

# VAE model = encoder + decoder

# build encoder model
# Capa de entrada
inputs = Input(shape=input_shape, name='encoder_input')
x = Flatten()(inputs)
# Capa intermedia 
h_x = Dense(intermediate_dim, activation='tanh')(x)

# dos capas de salida, media y varianza
z_mean = Dense(latent_dim, name='z_mean')(h_x)
z_log_var = Dense(latent_dim, name='z_log_var')(h_x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
# Capa de entrada
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# Capa intermedia
h_z = Dense(intermediate_dim, activation='tanh')(latent_inputs)
# Capa de salida, misma dimensión que la entrada del encoder
x_ = Dense(settings['seq_length'] * settings['variables'], activation='tanh')(h_z)
outputs = Reshape((settings['seq_length'], settings['variables'], 1))(x_)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


#%%
time_start = time.time()
if __name__ == '__main__':
    
#    mmd = K.mean(MMD2(inputs.numpy(), outputs.numpy(), sigma))
    reconstruction_loss = tf.reduce_mean(mse(inputs, outputs))
    reconstruction_loss *= (settings['seq_length'] * settings['variables'])
    #reconstruction_loss = BinaryCrossentropy(inputs, outputs)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    lr_schedule = optimizers.schedules.ExponentialDecay(
    settings['learning_rate'],
    decay_steps=settings['decay_step'],
    decay_rate=settings['decay_rate'],
    staircase=True)
    
    opt = optimizers.Adam(learning_rate=lr_schedule)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    VAE = vae.fit(samp,
            batch_size=20,
            epochs=epochs)


#%%
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)



data_exp_now["training_time"] = time.time() - time_start

vae.save(".//models//vae_model_"+date_exp+".h5")

data_exp_now = pd.DataFrame(data=data_exp_now, index=[0])

data_exp_hist = pd.read_csv('.//experiments_settings//experiments_vae.csv')
data_exp_hist = data_exp_hist.append(data_exp_now, ignore_index=True)
data_exp_hist.to_csv('.//experiments_settings//experiments_vae.csv', index=False )

#%%
fig = plt.figure()
plt.plot(VAE.history['loss'], color='g')
plt.legend(['los_vae'])
plt.xlabel('épocas')
plt.grid()
fig.savefig('results/loss_vae'+repr(date_exp)+'.pdf')
#np.save('./results/losses_vae'+repr(date_exp)+".npy", np.array(VA




#%%
sampts, labelsts, scalerts, columns_namests, timestampts, length_datats = samples.get_data(settings["seq_length"],
                                                                   settings["seq_step"],
                                                                   settings["aggregate"],
                                                                   scaler=None, path=settings["data_load_from"],
                                                                   scaler_type=settings["scaler_type"],
                                                                   start_time=None,
                                                                   name_scaler=date_exp)


#%%
#ejemplo

#scaler_filename = settings['path_scaler']
#scaler = joblib.load(scaler_filename) 

# --- get data, split --- #
# samples, pdf, labels = data_utils.get_samples_and_labels(settings)

import data_utils

samples_out, labels_out, scaler, columns_names, time_out = data_utils.get_data(settings["seq_length"], settings["seq_step"], settings["aggregate"], scaler=scaler, path='.//data//test//')
num_features = samples_out.shape[-1]

print('Data train shape:', samples_out.shape)

#%%

for k in range(samples_out.shape[0]):
    test = samples_out[k]
    test = np.reshape(test,(1, test.shape[0], test.shape[1], 1))
    test = np.repeat(test, settings['batch_size'], axis=0)
    
    rec = vae.predict(test)
    
    error_batch = []
    for i in range(test.shape[0]):
        real = np.reshape(test[i], (test.shape[1], test.shape[2]))
        rec_aux = np.reshape(rec[i], (rec.shape[1], rec.shape[2]))
        
        #for j in range(real.shape[1]):
        error2 = (real-rec_aux)**2   
        error2_mean = np.mean(error2, axis=1)
        
        error_batch.append(error2_mean)
    
    error_batch = np.array(error_batch)
    error_sample = np.min(error_batch, axis=0)
    
    if k == 0:
        pred = error_sample
        true = labels_out[k]
        
    else:
        pred = np.concatenate((pred, error_sample))
        true = np.concatenate((true, labels_out[k]))
#test_data = pd.read_csv('.//data//test//SWaT_Dataset_Attack_v0.csv')

#%%
#Evaluación

from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, precision_recall_curve, confusion_matrix

umbrales = np.arange(0.01,5,0.01)

F1 = []
ACC = []

for u in umbrales:
    
    score_aux = pred.copy()
    score_aux[score_aux < u] = 0
    score_aux[score_aux >= u] = 1
    F1.append(f1_score(true.astype(int), score_aux.astype(int)))
    ACC.append(accuracy_score(true.astype(int), score_aux.astype(int)))

F1 = np.array(F1)
ACC = np.array(ACC)

print(np.max(F1))
print(np.max(ACC))

#%%



pred_norm = pred/np.max(pred)


#%%

fpr, tpr, thresholds = roc_curve(true, pred_norm)           

fig = plt.figure()
plt.fill_between( fpr, tpr, color="skyblue", alpha=0.2)
plt.plot(fpr, tpr, color="Slateblue", alpha=0.6)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC (área = %0.2f)' % roc_auc_score(true, pred_norm))
plt.grid()
plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.6)
fig.savefig('results/roc)_vae'+settings['data']+'.pdf')

#%%

precision, recall, thresholds = precision_recall_curve(true, pred_norm)

fig = plt.figure()
plt.fill_between(recall, precision,  color="orange", alpha=0.2)
plt.plot(recall, precision, color="darkorange", alpha=0.6)

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall')
plt.grid()

#plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

fig.savefig('results/pre-rec_vae'+settings['data']+'.pdf')


#%%

#%%

y_pred  = pred.copy()
y_pred[y_pred < umbrales[np.argmax(F1)]] = 0
y_pred[y_pred >= umbrales[np.argmax(F1)]] = 1


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(true.astype(int), y_pred.astype(int), normalize = None)


cm_display = ConfusionMatrixDisplay(cm,  display_labels=['Normales', 'Anómalos']).plot().figure_
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas')
#plt.imsave('results/conf-matr'+settings['data']+'.pdf', cm_display)
#fig.savefig('results/conf-matr'+settings['data']+'.pdf')
cm_display.savefig('results/conf_matrix_vae.pdf')


#%%

first = datetime(2020, 1, 4, 0, 0, 0)
delta = timedelta(seconds = int(settings['aggregate']))
dates = list((first + delta * x for x in range(settings['seq_length'])))

data_gen = pd.DataFrame(samples_out[500], index = dates, columns = columns_names[1:-1])

label_gen = pd.DataFrame(labels_out[500], index = dates, columns = ['labels'])
#print(roc_auc_score(label.astype(int), 1 - sccore))
#print(f1_score(label.astype(int), (1 - sccore).round()))
#print(accuracy_score(label.astype(int), (1 - sccore).round()))
#%%
test = samples_out[500]
test = np.reshape(test,(1, test.shape[0], test.shape[1], 1))
test = np.repeat(test, settings['batch_size'], axis=0)

rec = vae.predict(test)
rec_aux = np.reshape(rec[4], (rec.shape[1], rec.shape[2]))

data_rec = pd.DataFrame(rec_aux, index = dates, columns = columns_names[1:-1])


#%%

fig = data_gen.plot(legend=False, ylim=[-1, 1.5]).get_figure()

fig.savefig('results/ejemplo_vae_real.pdf')

fig = data_rec.plot(legend=False, ylim=[-1, 1.5]).get_figure()

fig.savefig('results/ejemplo_vae_rec.pdf')


fig = label_gen.plot().get_figure()

fig.savefig('results/ejemplo_vae_lab.pdf')


#%%
# error2 = (samples_out[0]-rec_aux)**2   
# error2_mean = np.mean(error2, axis=1)
# error_df = pd.DataFrame(error2_mean, index = dates, columns = ['error'])

test = samples_out[500]
test = np.reshape(test,(1, test.shape[0], test.shape[1], 1))
test = np.repeat(test, settings['batch_size'], axis=0)

rec = vae.predict(test)

error_batch = []
for i in range(test.shape[0]):
    real = np.reshape(test[i], (test.shape[1], test.shape[2]))
    rec_aux = np.reshape(rec[i], (rec.shape[1], rec.shape[2]))
    
    #for j in range(real.shape[1]):
    error2 = (real-rec_aux)**2   
    error2_mean = np.mean(error2, axis=1)
    
    error_batch.append(error2_mean)

error_batch = np.array(error_batch)
error_sample = np.min(error_batch, axis=0)


error_df = pd.DataFrame(error_sample, index = dates, columns = [r'$error^2$'])

fig = error_df.plot(color='red').get_figure()

fig.savefig('results/ejemplo_vae_error.pdf')

# test = sampts[0]
# test = np.reshape(test,(1, test.shape[0], test.shape[1], 1))

# rec = vae.predict(test)

# rec = np.reshape(rec, (rec.shape[1], rec.shape[2]))

# #VAE.history['loss']))

# #%%
# #vae2 = load_model('.//models//vae_model_06-08-2020 20_01_04.h5')

# #%%
# callCost = pd.read_csv('.//data//train//callCost.csv')
# #callCost_gen['time'] = callCost_gen['Unnamed: 0']
# callCost = callCost.drop(columns='labels')
# callCost['time'] = pd.to_datetime(callCost.time) #loc
# callCost = callCost.set_index('time')
# callCost = callCost.resample('1200S').mean()

# #scaler = MinMaxScaler(feature_range=(-1, 1))
# #scaler.fit(callCost.values)
# callCost_norm = scaler.transform(callCost.values)
# callCost = pd.DataFrame(callCost_norm, index=callCost.index, columns=callCost.columns)

# #%%
# start_time=pd.to_datetime('2020/03/31 07:00:00')
# end_time=pd.to_datetime('2020/04/01 06:59:00')
# #callCost_gen.plot()
# #callCost[start_time:end_time].plot(ylim=[-1,1], legend=False)
# #plt.grid()
# array_df = callCost[start_time:end_time]
# array = array_df.values


# array_aux = np.reshape(array, (1, array.shape[0], array.shape[1],1))
# #x_i = X[i]
# #x_i = np.repeat(array, 72, axis=0)


# #%%
# recons  = recontrVAE(array_df, vae)




# #%%
# predic = vae.predict(array_aux)

# #%%
# predic = np.reshape(predic, (predic.shape[1], predic.shape[2]))
# predic = pd.DataFrame(predic, index = array_df.index, columns=columns_names[1:-1])

# #%%
# comparation = pd.concat((array_df, predic), axis=1)
# comparation.plot(ylim=[-1,1], legend=False)
# plt.grid()

# #%%
# def recontrVAE(array_df, model):
#     array = array_df.values
#     array_aux = np.reshape(array, (1, array.shape[0], array.shape[1],1))
#     corr = []
#     predic_array = []
#     for i in range(settings['batch_size']):
#         predic = model.predict(array_aux)
#         predic = np.reshape(predic, (predic.shape[1], predic.shape[2]))
#         corr.append(np.correlate(array[:,0], predic[:,0]))
#         predic_array.append(predic)
        
#     index_max_corr = np.argmax(np.array(corr)) 
#     predic = predic_array[index_max_corr]
#     predic = pd.DataFrame(predic, index = array_df.index, columns=columns_names[1:-1])
    
#     return predic

# #%%
# first = datetime(2020, 1, 13, 0, 0, 0)
# delta = timedelta(seconds = int(settings['aggregate']*settings['seq_length']))
# #dates = list((first + delta * x for x in range(settings['seq_length'])))
# serie_callCost = pd.DataFrame()
# serie_callCost_VAE = pd.DataFrame()

# start_time=first
# end_time=first + delta - timedelta(seconds = int(settings['aggregate']))

# #%%
# for day in range(7):
#     array = callCost[start_time:end_time]
#     serie_callCost = pd.concat((serie_callCost, array))    
#     recon = recontrVAE(array, vae)
#     serie_callCost_VAE = pd.concat((serie_callCost_VAE, recon))
    
#     start_time = start_time + delta
#     end_time = end_time + delta
    
# #%%
# comparation = pd.concat((serie_callCost, serie_callCost_VAE), axis=1)
# comparation.plot(ylim=[-1,1])
# plt.grid()
# #%%

# serie_callCost_VAE.plot(ylim=[-1,1])
# #%%
# serie_callCost_VAE.to_csv('prueba_movistar10_VAE.csv')
