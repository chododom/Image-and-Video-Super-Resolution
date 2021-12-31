# inspired by: https://github.com/sevakon/unet-keras/blob/master/model/utils.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, mean_squared_error
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
import os
import random
import warnings


#################### Building blocks ####################

def input_tensor(input_size):
    x = Input(input_size)
    return x

# function that defines one convolutional layer with certain number of filters
def single_conv(input_tensor, n_filters, kernel_size):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), activation = 'sigmoid')(input_tensor)
    return x

# function that defines two sequential 2D convolutional layers with certain number of filters
def double_conv(input_tensor, n_filters, kernel_size = 3):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# function that defines 2D transposed convolutional layer
def deconv(input_tensor, n_filters, kernel_size = 3, stride = 2):
    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides = (stride, stride), padding = 'same')(input_tensor)
    return x

# function that defines max pooling layer with pool size 2 and applies dropout
def pooling(input_tensor, dropout_rate = 0.1):
    x = MaxPooling2D(pool_size = (2, 2))(input_tensor)
    x = Dropout(rate = dropout_rate)(x)
    return x
    
#################### SR-U-Net ####################

def unet(img_size):
    input_dims = (img_size, img_size, 3)
    n_filters = 64
    
    input_layer = input_tensor(input_dims)
    
    # contracting path
    conv1 = double_conv(input_layer, n_filters * 1)
    pool1 = pooling(conv1)
    
    conv2 = double_conv(pool1, n_filters * 4)
    pool2 = pooling(conv2)
    
    conv3 = double_conv(pool2, n_filters * 8)
    pool3 = pooling(conv3)
    
    
    # middle
    conv4 = double_conv(pool3, n_filters * 8)
    
    
    # expansive path
    up5 = deconv(conv4, n_filters * 8)
    up5 = concatenate([conv3, up5])
    conv5 = double_conv(up5, n_filters * 8)
    
    up6 = deconv(conv5, n_filters * 4)
    up6 = concatenate([conv2, up6])
    conv6 = double_conv(up6, n_filters * 4)
    
    up7 = deconv(conv6, n_filters * 2)
    up7 = concatenate([conv1, up7])
    conv7 = double_conv(up7, n_filters * 2)
    
    # upscale on top of U-Net to get higher resolution than input
    up8 = deconv(conv7, n_filters * 1)
    conv8 = double_conv(up8, n_filters * 1)
    
    # output layer
    output = Conv2D(3,(1,1), padding="same")(conv8)
    
    # if the input is 128 x 128, then we perform one more expansion to get to 256 x 256
    if img_size == 128:
        up9 = deconv(conv8, n_filters * 1)
        conv9 = double_conv(up9, n_filters * 1)
        output = Conv2D(3,(1,1), padding="same")(conv9)
        
    model = Model(input_layer, output, name='SR-U-Net')
    return model