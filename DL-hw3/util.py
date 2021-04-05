import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import time

def read_dataset(height,width,batch_size):

	dataset = keras.preprocessing.image_dataset_from_directory(
    "celeba_gan/", 
    label_mode=None, 
    image_size=(128, 128),
    batch_size=batch_size,
    shuffle=True)
	dataset = dataset.map(lambda x: (x-127.5)/127.5).prefetch(2)
	return dataset

def generate_latent_points(SAMPLES):
    return tf.random.normal([SAMPLES,LATENT_DIM])

def generate_real_samples(samples):
    for batch in dataset.shuffle(5):
        return batch
        
def plot_figs(epoch):
    x_fake = [gan() for i in range(8)]
    for k in range(8):
        plt.subplot(2, 4, k+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(gan()))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    fig_name = 'save_figs/generated-image-'+str(epoch)
    plt.savefig(fig_name)

def plot_loss (gen_loss, dis_loss):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss - EPOCH ")
    plt.plot(gen_loss,label="Generator")
    plt.plot(dis_loss,label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def read_loss_file():
	with open('gen_loss.txt') as f:
    	g = f.read().splitlines()
	with open('dis_loss.txt') as f:
	    d = f.read().splitlines()
	return np.array(g, dtype ='float32'),np.array(d, dtype ='float32')
