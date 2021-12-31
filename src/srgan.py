# inspired by: https://github.com/deepak112/Keras-SRGAN/

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate, Dense, UpSampling2D, Flatten, add
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, mean_squared_error
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
import numpy as np
import os
import random
import warnings
from tqdm import tqdm

from utils import plot_comparison

# residual convolutional block for G
def res_block_gen(model, kernal_size, filters, strides):
    gen = model
    block = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    block = BatchNormalization(momentum = 0.5)(block)
    block = PReLU(alpha_initializer='zeros', shared_axes=[1,2])(block)
    block = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(block)
    block = BatchNormalization(momentum = 0.5)(block)
        
    block = add([gen, block])
    
    return block
    
# upsamling block to increase resolution  
def up_sampling_block(model, kernal_size, filters, strides):
    up = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    up = UpSampling2D(size = 2)(up)
    up = LeakyReLU(alpha = 0.2)(up)
    
    return up

# convolutional block for D
def discriminator_block(model, filters, kernel_size, strides):
    block = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    block = BatchNormalization(momentum = 0.5)(block)
    block = LeakyReLU(alpha = 0.2)(block)
    
    return block

# G architecture as described by https://arxiv.org/pdf/1609.04802.pdf
def get_generator(input_shape):
    gen_input = Input(shape=input_shape)
    
    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    
    gen_model = model
    
    # stack 16 residual blocks
    for i in range(16):
        model = res_block_gen(model, 3, 64, 1)
    
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = add([gen_model, model])
    
    # upsample to increase resolution
    model = up_sampling_block(model, 3, 256, 1)
    
    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    
    generator_model = Model(inputs=gen_input, outputs=model)
    return generator_model

# D architecture as described in https://arxiv.org/pdf/1609.04802.pdf
def get_discriminator(input_shape):
    disc_input = Input(shape=input_shape)
    
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(disc_input)
    model = LeakyReLU(alpha = 0.2)(model)
    
    model = discriminator_block(model, 64, 3, 2)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 2)
    model = discriminator_block(model, 256, 3, 1)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 512, 3, 1)
    model = discriminator_block(model, 512, 3, 2)
        
    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha = 0.2)(model)
       
    # one neuron output to signify probability of real sample
    model = Dense(1)(model)
    model = Activation('sigmoid')(model) 
        
    discriminator_model = Model(inputs=disc_input, outputs=model)
        
    return discriminator_model
    
# setup GAN architecture
def get_gan_network(discriminator, shape, generator, optimizer, loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    
    sr_image = generator(gan_input)
    gan_output = discriminator(sr_image)
    
    gan = Model(inputs=gan_input, outputs=[sr_image, gan_output])
    
    # weighted sum of losses
    gan.compile(loss=[loss, 'binary_crossentropy'],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan
    
# training loop for the SR-GAN
def train_gan(img_size, target_size, epochs, batch_size, batch_cnt, train_gen, loss, save_dir, desc):
    gen = get_generator((img_size, img_size, 3))
    disc = get_discriminator((target_size, target_size, 3))
   
    optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    gen.compile(loss=loss, optimizer=optimizer)
    disc.compile(loss='binary_crossentropy', optimizer=optimizer)

    gan = get_gan_network(disc, (img_size, img_size, 3), gen, optimizer, loss)

    loss_file = open(os.path.join(save_dir, 'losses' + desc + '.txt'), 'w+')
    loss_file.close()

    # epoch loop
    for e in range(1, epochs + 1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        
        # batch loop
        for _ in tqdm(range(batch_cnt)):

            lr_batch, hr_batch = next(train_gen)
            generated_sr = gen.predict(lr_batch)

            # add random noise of up to 20 % to the 0.0 fake labels and deduct up to 20 % noise from the 1.0 % true labels
            real_data_Y = np.ones(len(hr_batch)) - np.random.random_sample(len(hr_batch))*0.2
            fake_data_Y = np.random.random_sample(len(generated_sr))*0.2
            
            disc.trainable = True
            
            # train D
            d_loss_real = disc.train_on_batch(hr_batch, real_data_Y)
            d_loss_fake = disc.train_on_batch(generated_sr, fake_data_Y)
            disc_loss = 0.5 * np.add(d_loss_fake, d_loss_fake)
            
            disc.trainable = False            
            
            lr_batch, hr_batch = next(train_gen)
            
            # train G
            gan_Y = np.ones(len(hr_batch)) - np.random.random_sample(len(hr_batch))*0.2
            gan_loss = gan.train_on_batch(lr_batch, [hr_batch, gan_Y])

        print("discriminator_loss : %f" % disc_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)

        loss_file = open(os.path.join(save_dir, 'losses.txt'), 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, disc_loss))
        loss_file.close()

        gen.save(os.path.join(save_dir, desc + '_G_64_to_128__%d.h5' % e))
        disc.save(os.path.join(save_dir, desc + '_D_64_to_128__%d.h5' % e))
        
        plot_comparison(gen, train_gen)
