import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
from model import GAN
import time
from util import read_dataset, generate_latent_points, generate_real_samples, plot_figs, read_loss_file

HEIGHT = 128
WIDTH = 128
CHANNELS = 3
SAMPLES = 32
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 500

dataset = read_dataset(HEIGHT,WIDTH,BATCH_SIZE)

def train(bat_per_epo):
    for epoch in range(EPOCHS):
        start_time = time.time()
        d_loss1,g_loss1=0,0
        for i in range(bat_per_epo):
            losses = gan.train_step(generate_latent_points(SAMPLES),generate_real_samples(SAMPLES))
            d_loss1+=losses['disc_loss']
            g_loss1+=losses['gen_loss']
        print('>%d, discriminator =%.3f, generator loss=%.3f time taken= %.2fs' %
                (epoch+1, d_loss1/bat_per_epo, g_loss1/bat_per_epo,(time.time()-start_time)))
        if epoch%10==0:
            plot_figs()
        if (epoch+1)%25==0:
            tf.saved_model.save(gan,"gan_model3/")
        with open('gen_loss.txt', 'a') as f:
        	f.write("%0.3f\n" % (g_loss1/bat_per_epo))
        with open('dis_loss.txt', 'a') as f:
        	f.write("%0.3f\n" % (d_loss1/bat_per_epo))

gan = GAN(LATENT_DIM)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8),
    g_optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))
train(25)

gen_loss,dis_loss = read_loss_file()

plot_loss(gen_loss,dis_loss)