import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import time

class GAN(tf.Module):
    def __init__(self,LATENT_DIM):
        super(GAN, self).__init__()
        self.kernel_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.kernel_const = tf.keras.constraints.max_norm(1.0)
        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()
        self.SAMPLES = SAMPLES
        self.latent_dim = LATENT_DIM
        self.resize = tf.keras.layers.experimental.preprocessing.Resizing(218,178,'lanczos3')

    def compile(self,  g_optimizer,d_optimizer, loss_fn):
        self.gen_optimizer = g_optimizer
        self.disc_optimizer = d_optimizer
        self.loss_fn =loss_fn
        
    
    def get_generator(self):
        layer1 = keras.Sequential([
            tf.keras.layers.Dense(8*8*256,input_shape=(LATENT_DIM,),
                                           kernel_initializer= self.kernel_init, use_bias=False,kernel_constraint=self.kernel_const),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Reshape(target_shape=(8,8,256)),

        ])
        layer2= keras.Sequential([
            tf.keras.layers.Conv2DTranspose(256, (4,4), padding='same', use_bias=False,strides=(2,2),
                                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.LeakyReLU(alpha=0.2),

        ])
        layer3= keras.Sequential([
            tf.keras.layers.Conv2DTranspose(128, (4,4), padding='same', use_bias=False,strides=(2,2),
                                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.LeakyReLU(alpha=0.2),

        ])
        layer4= keras.Sequential([
            tf.keras.layers.Conv2DTranspose(128, (4,4), padding='same', use_bias=False,strides=(2,2),
                                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.LeakyReLU(alpha=0.2),

        ])
        out = tf.keras.layers.Conv2DTranspose(3, (4,4), padding='same', use_bias=False,strides=(2,2),
                                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const,activation='tanh')
        return keras.Sequential([layer1,layer2,layer3,layer4,out])

    def get_discriminator(self):
        layer1 = keras.Sequential([
            tf.keras.layers.Input(shape=(HEIGHT, WIDTH, CHANNELS)),
            tf.keras.layers.Conv2D(256, (4,4), padding='same', strides=(2,2),
                                            kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])
        layer2 = keras.Sequential([
            tf.keras.layers.Conv2D(256, (4,4), padding='same', strides=(2,2),
                                            kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])
        layer3 = keras.Sequential([
            tf.keras.layers.Conv2D(128, (4,4), padding='same', strides=(2,2),
                                            kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])
        layer4= keras.Sequential([
            tf.keras.layers.Conv2D(128, (4,4), padding='same', strides=(2,2),
                                            kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])
        out = keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1)  
        ])
        return keras.Sequential([layer1,layer2,layer3,layer4,out])
        
    @tf.function#(input_signature=[tf.TensorSpec([SAMPLES, HEIGHT,WIDTH,3 ], tf.float32)])
    def train_step(self,noise,images):
        #noise = generate_latent_points(SAMPLES)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise,training = True)
            real_output = self.discriminator(images)
            generated_output = self.discriminator(generated_images)
            gen_loss = self.generator_loss(generated_output)
            disc_loss = self.discriminator_loss(real_output, generated_output)
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return {'gen_loss':gen_loss,
                'disc_loss':disc_loss}
    
    @tf.function(input_signature=[])
    def __call__(self):
        self.generator.training=False
        x = self.generator(tf.random.normal([1,LATENT_DIM]))
        x = self.resize(x)
        self.generator.training=True
        return tf.reshape(x,shape=(218,178,3))

    def discriminator_loss(self,real_logits, fake_logits):
        y = self.smooth_positive_labels(self.noisy_labels(np.ones((SAMPLES, 1)))).astype('float32')
        y_f= self.smooth_negative_labels(self.noisy_labels(np.zeros((SAMPLES,1)))).astype('float32')
        real_loss = self.loss_fn(tf.convert_to_tensor(y), real_logits)
        fake_loss = self.loss_fn(tf.convert_to_tensor(y_f), fake_logits)
        return real_loss + fake_loss

    def generator_loss(self,fake_logits):
        return self.loss_fn(tf.ones_like(fake_logits),fake_logits)
    
    def noisy_labels(self,y, p_flip=0.08):
        n_select = int(p_flip * y.shape[0])
        flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
        y[flip_ix] = 1 - y[flip_ix]
        return y

    def smooth_positive_labels(self,y):
        return y - 0.3 + (np.random.random(y.shape) * 0.5)
    
    def smooth_negative_labels(self,y):
        return y + np.random.random(y.shape) * 0.3