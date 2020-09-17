# -*- coding: utf-8 -*-
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

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from MMD import MMD2
import numpy as np
import samples
import json
import time
import random
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

tf.keras.backend.set_floatx('float64')

#%%
#date of the experiment
date_exp = str(datetime.now().strftime("%d-%m-%Y %H_%M_%S"))

#settings
settings_path = './settings/' + 'train_gan_swat' + '.txt'
print('Loading settings from', settings_path)
settings = json.load(open(settings_path, 'r'))

# --- get data, split --- #
samples, labels, scaler, columns_names, timestamp, length_data = samples.get_data(settings["seq_length"],
                                                                   settings["seq_step"],
                                                                   settings["aggregate"],
                                                                   scaler=None, path=settings["data_load_from"],
                                                                   scaler_type=settings["scaler_type"],
                                                                   start_time=None,
                                                                   name_scaler=date_exp)
#sigma value to calculate MMD
sigma = np.std(samples)
print('Data train shape:', samples.shape)

#Experiment settings
data_exp_now = {'date': date_exp, 
                'data': settings['data'],
                'path': settings["data_load_from"], 
                'data_size': length_data,
                'num_muestras_train': samples.shape[0], 
                'seq_length': settings['seq_length'], 
                'seq_step': settings["seq_step"], 
                'aggregate': settings["aggregate"], 
                'variables': settings["variables"],
                'hidden_units_g': settings['hidden_units_g'], 
                'hidden_units_d': settings['hidden_units_d'], 
                'latent_dim': settings['latent_dim'], 
                'learning_rate': settings['learning_rate'], 
                'decay_step': settings['decay_step'], 
                'decay_rate': settings['decay_rate'], 
                'batch_size': settings['batch_size'], 
                'num_epochs': settings['num_epochs'], 
                'label_max': settings['label_max'], 
                'label_min': settings['label_min'], 
                'scaler_type': settings['scaler_type'],
                'path_scaler': "scaler_"+date_exp+".save",
                'scaler': scaler,
                'path_dis_model': ".//models//dis_model_"+date_exp+".h5",
                'path_gen_model': ".//models//gen_model_"+date_exp+".h5",
                'training_time': 0
                }

#%%
# build the generator
def make_generator(settings):
    model_g = Sequential(name="generator")
    model_g.add(LSTM(settings['hidden_units_g'], batch_input_shape=(settings['batch_size'], settings['seq_length'], settings['latent_dim']), return_sequences=True, stateful=True, name='hidden_layer_g'))
    model_g.add(Dense(settings["variables"], activation='tanh', name='output_g'))
    #, kernel_initializer='zeros'
    return model_g

#%%

# build the discriminator
def make_discriminator(settings):
    model_d = Sequential(name="discriminator")
    model_d.add(LSTM(settings['hidden_units_d'], batch_input_shape=(settings['batch_size'], settings['seq_length'], settings["variables"]), return_sequences=True, stateful=True, name='hidden_layer_d'))
    model_d.add(Dense(1, activation= 'sigmoid', name='output_d'))
    #, kernel_initializer='zeros
    # 'sigmoid'
    # stateful=True,
    return model_d

#%%
# define the loss function for the discriminator
cross_entropy_d = BinaryCrossentropy(reduction= Reduction.NONE)

def discriminator_loss(real_output, fake_output):
    # real_output is the prediction of the discriminator for a batch of real samples
    # fake_output is the same but for samples from the generator
    real_loss = cross_entropy_d(random.uniform(settings['label_min'], settings['label_max'])*tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_d(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss)
    return total_loss, real_loss, fake_loss

#%%
# define the loss function for the generator
cross_entropy_g = BinaryCrossentropy(reduction= Reduction.NONE)

def generator_loss(fake_output):
    return cross_entropy_g(random.uniform(settings['label_min'],settings['label_max'])*tf.ones_like(fake_output), fake_output)

#%%
 
# learning rate for both optimizers
lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
    settings['learning_rate'] ,
    decay_steps=settings['decay_step']*settings['G_rounds'],
    decay_rate=settings['decay_rate'],
    staircase=True)

lr_schedule_d = tf.keras.optimizers.schedules.ExponentialDecay(
    settings['learning_rate'] ,
    decay_steps=settings['decay_step']*settings['D_rounds'],
    decay_rate=settings['decay_rate'],
    staircase=True)

generator_optimizer = tf.keras.optimizers.Adam(lr_schedule_g)
discriminator_optimizer = tf.keras.optimizers.SGD(lr_schedule_d)

#%%
gen_loss_mean = tf.keras.metrics.Mean(name='gen_loss_mean')
disc_loss_mean = tf.keras.metrics.Mean(name='disc_loss_mean')
#disc_loss_real_mean = tf.keras.metrics.Mean(name='disc_loss_real_mean')
#disc_loss_fake_mean = tf.keras.metrics.Mean(name='disc_loss_fake_mean')
MMD_mean = tf.keras.metrics.Mean(name='MMD')
#%%
def train_gen_step(settings, generator, discriminator):  
    
    # batch_size samples from p(z)                                    
    noise = tf.random.normal([settings['batch_size'], settings['seq_length'], settings['latent_dim']])

    with tf.GradientTape() as gen_tape:
        
        generated_samples = generator(noise)
        fake_output = discriminator(generated_samples)
        gen_loss = generator_loss(fake_output)
        
    gen_loss_mean(gen_loss)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

#%%
def train_dis_step(samples, settings, generator, discriminator):  
    
    noise = tf.random.normal([settings['batch_size'], settings['seq_length'], settings['latent_dim']])
    with tf.GradientTape() as disc_tape:
        generated_samples = generator(noise)
        real_output = discriminator(samples)
        fake_output = discriminator(generated_samples)
        disc_loss, disc_real_loss, disc_fake_loss  = discriminator_loss(real_output, fake_output)
    

    disc_loss_mean(disc_loss)
#    disc_loss_real_mean(disc_real_loss)
#    disc_loss_fake_mean(disc_fake_loss)
    MMD_mean(MMD2(samples, generated_samples.numpy(), sigma))
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#%%
def train_GAN(epochs, samples_out, settings, generator, discriminator):  
    results = []
#    template = 'Epoch {:03d}: time {:.1f} sec, gen_loss {:.4f}, disc_loss {:.4f}, disc_lossReal {:.4f}, disc_lossFake {:.4f}'
    template = 'Epoch {:03d}: time {:.1f} sec, gen_loss {:.4f}, disc_loss {:.4f}, MMD {:.4f}'
 
    cont_image = 0
    for epoch in range(epochs):
        start = time.time()
    
        for ix_batch in range(0, len(samples_out) // settings['batch_size']):
            samples_batch = samples_out[ix_batch*settings['batch_size']:(ix_batch+1)*settings['batch_size'],:,:]
            for d in range(settings['D_rounds']):
                train_dis_step(samples_batch, settings, generator, discriminator)
            for g in range(settings['G_rounds']):
                train_gen_step(settings, generator, discriminator)
  
#==========evolution of the generation======================================
        if cont_image == epoch:
            noise = tf.random.normal([settings['batch_size'], settings['seq_length'], settings['latent_dim']])
            gen_samples_exam = generator(noise)
            data_gen = pd.DataFrame(gen_samples_exam[settings['batch_size']//2,:,:].numpy(), index = pd.to_datetime(timestamp[0][0:settings['seq_length']]), columns = columns_names[1:-1])
            fig = data_gen.plot(ylim=[-1,1], legend=False).get_figure()
            plt.title('Epoch: '+repr(cont_image))
            fig.savefig('generated_series/epoch'+repr(cont_image)+'.png')
            plt.close()
            cont_image += settings["epochs_to_generation_view"]
            
#=============================================================================        
        print(template.format(epoch + 1, time.time()-start, gen_loss_mean.result(), disc_loss_mean.result(), MMD_mean.result()))
        results.append([gen_loss_mean.result(), disc_loss_mean.result(), MMD_mean.result()])
        
    
    return results

#%%   
generator = make_generator(settings) 
discriminator = make_discriminator(settings)

#generator = load_model('.//models//gen_model_13-08-2020 19_09_56.h5')
#discriminator = load_model('.//models//dis_model_13-08-2020 19_09_56.h5')
#%%
time_start = time.time()
losses = train_GAN(settings['num_epochs'], samples, settings, generator, discriminator)
data_exp_now["training_time"] = time.time() - time_start
#%%
data_exp_now["training_time"] = 0
#%%
discriminator.save(".//models//dis_model_"+date_exp+".h5")
generator.save(".//models//gen_model_"+date_exp+".h5")


#%%
discriminator.save(".//models//dis_model_"+date_exp+".h5")
generator.save(".//models//gen_model_"+date_exp+".h5")

data_exp_now = pd.DataFrame(data=data_exp_now, index=[0])

data_exp_hist = pd.read_csv('.//experiments_settings//experiments.csv')
data_exp_hist = data_exp_hist.append(data_exp_now, ignore_index=True)
data_exp_hist.to_csv('.//experiments_settings//experiments.csv', index=False )


#%%
losses_arr = np.array(losses)
fig = plt.figure()
plt.plot(losses_arr[:,0])
plt.plot(losses_arr[:,1])

plt.grid()
plt.legend(['loss_gen', 'loss_dis'])
fig.savefig('results/losses'+date_exp+'.pdf')

fig1 = plt.figure()
plt.plot(losses_arr[:,2])
plt.grid()
plt.legend(['MMD'])
fig1.savefig('results/MMD'+date_exp+'.pdf')

np.save('./results/losses_MMD_gan'+date_exp+".npy", losses_arr)

