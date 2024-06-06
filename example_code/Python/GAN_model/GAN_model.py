import sys, os
from pathlib import Path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys, os
import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from tensorflow.keras import Model
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib
from PIL import Image

###### components ######
#helper functions do not change
kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.)

def swish(x):
    return tf.keras.activations.swish(x)

def relu(x):
    return tf.keras.activations.relu(x)

def leakyrelu(x):
    return tf.keras.layers.LeakyReLU(.2)(x)

def insnorm(x):
    return tfa.layers.InstanceNormalization(axis=-1)(x)

def pixelnorm(x):
    return x/tf.math.sqrt(tf.reduce_mean(x**2+(10e-8), axis = 3, keepdims=True))

def batchnorm(x):
    return layers.BatchNormalization(axis=-1)(x)

#minibatch standard deviation as dscribed in Progressive GAN Tero Karras et al. https://github.com/tkarras/progressive_growing_of_gans
def minibatch_stddev_layer(x, group_size=4):
    # Minibatch must be divisible by (or smaller than) group_size.
    group_size = tf.minimum(group_size, tf.shape(x)[0]) 
    # [NCHW]  Input shape.    
    s = x.shape
    # [GMCHW] Split minibatch into M groups of size G.                                             
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   
    # [GMCHW] Cast to FP32.
    y = tf.cast(y, tf.float32)  
    # [GMCHW] Subtract mean over group.                            
    y -= tf.reduce_mean(y, axis=0, keepdims=True)   
    # [MCHW]  Calc variance over group.        
    y = tf.reduce_mean(tf.square(y), axis=0)   
    # [MCHW]  Calc stddev over group.             
    y = tf.sqrt(y + 1e-8)          
    # [M111]  Take average over fmaps and pixels.                         
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True) 
    # [M111]  Cast back to original data type.     
    y = tf.cast(y, x.dtype) 
    # [N1HW]  Replicate over group and pixels.                                
    y = tf.tile(y, [group_size, s[1], s[2], 1])             
    return tf.concat([x, y], axis=3)

def upsample_d(x, factor=2):
    return layers.UpSampling2D(
                size=(factor, factor), interpolation='nearest'
            )(x)

def upsample(x, filters, kernel_size=(3, 3), padding="same", factor=2):
    return layers.Conv2DTranspose(filters, kernel_size,
                                    strides=(factor, factor), padding=padding)(x)

def avgpooling2D(x,factor=2):
    return layers.AveragePooling2D(pool_size=(2, 2),strides=(factor, factor), padding='same')(x)


#Build custom tensorflow layers for equalised 2d convolution, equalised 2d transpose convolution and equalised dense layer
#scale the weights dynamically by the sqrt(gain/(kernalh*kernelw*numberchannels))
#apply this scaling as an input into convolution/dense function, do not scale the bias
#make sure to set the kernel initilization seed to 42 so results are consistent

class EqualizedConv2D(tf.keras.layers.Layer):
    def __init__(self, 
                filters, 
                kernel_size=(3,3), 
                strides=(1,1), 
                kernel_initializer=tf.initializers.RandomNormal(seed=42), 
                bias_initializer=tf.initializers.Zeros(), 
                gain=2,
                **kwargs):
        
        super(EqualizedConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.gain = gain

    def build(self, input_shape):
        *_, n_channels = input_shape
        fan_in = self.kernel_size[0]*self.kernel_size[1]*n_channels
        self.scale = tf.math.sqrt(self.gain/fan_in)
        
        self.w = self.add_weight(
                name='kernel',
                shape=(*self.kernel_size,
                            n_channels,
                            self.filters),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=tf.float32)

        self.b = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.float32)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "gain": self.gain,
        })
        return config
    
    def call(self, inputs, training=None):
        x = tf.nn.conv2d(inputs, filters=self.scale*self.w, strides=self.strides, padding = 'SAME')

        x = x + self.b

        return x

class EqualizedConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                kernel_size=(3,3),
                strides=(1,1), 
                kernel_initializer=tf.initializers.RandomNormal(seed=42),
                bias_initializer=tf.initializers.Zeros(),
                gain=2,
                **kwargs):
        
        super(EqualizedConv2DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.gain = gain

    def build(self, input_shape):
        *_, n_channels = input_shape
        fan_in = self.kernel_size[0]*self.kernel_size[1]*n_channels
        self.scale = tf.math.sqrt(self.gain/fan_in)
        
        self.w = self.add_weight(
                name='kernel',
                shape=(*self.kernel_size,
                            self.filters,
                            n_channels),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=tf.float32)

        self.b = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.float32)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "gain": self.gain,
        })
        return config

    def call(self, inputs, training=None):
        x = tf.nn.conv2d_transpose(inputs, filters=self.scale*self.w,
                                output_shape = (inputs.shape[1]*self.strides[0],inputs.shape[2]*self.strides[1]),
                                strides=self.strides,
                                padding = 'SAME')

        x = x + self.b

        return x

class EqualizedDense(layers.Layer):
    def __init__(self,
                units=1,
                kernel_initializer=tf.initializers.RandomNormal(seed=42),
                bias_initializer=tf.initializers.Zeros(),
                gain=2,
                **kwargs):
        
        super(EqualizedDense, self).__init__(**kwargs)

        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.gain = gain
        
        
    def build(self, input_shape):
        
        *_, n_channels = input_shape
        
        self.scale = tf.math.sqrt(self.gain/n_channels)
        
        self.w = self.add_weight(
                name='kernel',
                shape=(n_channels,
                        self.units),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=tf.float32)
        
        self.b = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.float32)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "gain": self.gain,
        })
        return config
    
    def call(self, inputs):
        return tf.matmul(inputs,self.scale*self.w) + self.b

#### # Test the custom layers
tf.random.set_seed(42)
input_shape = (1, 8, 8, 3)
x = tf.random.normal(input_shape)

equalized_conv2d = EqualizedConv2D(filters=16, kernel_size=(3, 3))
output_equalized_conv2d = equalized_conv2d(x)
print('conv2d_equal',output_equalized_conv2d.shape)
print('conv2d_equal',output_equalized_conv2d)

equalized_transposeconv2d = EqualizedConv2DTranspose(filters=16, kernel_size=(3, 3))
output_equalized_transposeconv2d = equalized_transposeconv2d(x)
print('transconv2d_equal',output_equalized_transposeconv2d.shape)
print('transconv2d_equal',output_equalized_transposeconv2d)

equalized_dense = EqualizedDense()
output_equalized_dense = equalized_dense(x)
print('dense_equal',output_equalized_dense.shape)
print('dense_equal',output_equalized_dense)

    
###### components ######

###### U-gen ######
# U generator initial block
def U_gen_bottom_init(
    act_func,
    norm_func,
    filters=512,
    kernel_init=kernel_init
):
    inputs = layers.Input(shape = (1,1,filters))
    x = EqualizedConv2DTranspose(filters,
                                kernel_size=(4,4),
                                strides=(4,4),
                                kernel_initializer=kernel_init)(inputs)
            
    x = act_func((x))
    x = EqualizedConv2D(filters, 
                        kernel_size=(3,3),           
                        strides=(1,1),
                        kernel_initializer=kernel_init)(x)
    
    x = norm_func(act_func(x))
    
    model = tf.keras.models.Model(inputs, [x,x])
    return model

#instantiate the U gen bottom initial block and print layer shapes
U_gen_bottom_init_layer_shape_lst = []
U_gen_bottom_init_fn = U_gen_bottom_init(leakyrelu, pixelnorm, 512, kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42)) 
for layer in U_gen_bottom_init_fn.layers:
    U_gen_bottom_init_layer_shape_lst.append(layer.output_shape)
print(U_gen_bottom_init_layer_shape_lst)

#define random tensor set see so consistent and feed through the previous instantiated block
#print output tensor and shape
x_U_gen_bottom_init = tf.random.normal((1,1, 1, 512))
output_U_gen_bottom_init = U_gen_bottom_init_fn(x_U_gen_bottom_init)
print(output_U_gen_bottom_init[0])
print(output_U_gen_bottom_init[0].shape)

def U_gen_bottom_add(
    act_func,
    norm_func,
    upsample_func,
    input_shape,
    filters,
    init_filters=512,
    kernel_init=kernel_init
):
    inputs = layers.Input(shape = (input_shape[0], input_shape[1], init_filters))
    
    upsample = upsample_func(inputs)
    
    x = norm_func(act_func(EqualizedConv2D(filters,
                                        kernel_size=(3,3), 
                                        strides=(1,1),
                                        kernel_initializer=kernel_init)(upsample)))
    
    x = norm_func(act_func(EqualizedConv2D(filters,
                                        kernel_size=(3,3), 
                                        strides=(1,1),
                                        kernel_initializer=kernel_init)(x)))
    
    model = tf.keras.models.Model(inputs, [x,upsample])
    return model 

#instantiate the U gen bottom additional blocks and print layer shapes
U_gen_bottom_add_fn_layer_shape_lst = []
U_gen_bottom_add_fn = U_gen_bottom_add(leakyrelu, pixelnorm,upsample_d, (4,4,512), 512, kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42)) 
for layer in U_gen_bottom_add_fn.layers:
    U_gen_bottom_add_fn_layer_shape_lst.append(layer.output_shape)
print(U_gen_bottom_add_fn_layer_shape_lst)

#define random tensor set see so consistent and feed through the previous instantiated block
#print output tensor and shape
x_U_gen_bottom_add = tf.random.normal((1,4, 4, 512))
U_gen_bottom_add_output = U_gen_bottom_add_fn(x_U_gen_bottom_add)
print(U_gen_bottom_add_output[0])
print(U_gen_bottom_add_output[1])
print(U_gen_bottom_add_output[0].shape)
print(U_gen_bottom_add_output[1].shape) 

def U_gen_top_init(
    act_func,
    filters=512,
    kernel_init=kernel_init
):
    inputs = layers.Input(shape = (4,4,filters))
    
    x = minibatch_stddev_layer(inputs, group_size=4)
    
    x = EqualizedConv2D(filters, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        kernel_initializer=kernel_init)(x)
    x = act_func(x)
        
    x = EqualizedConv2D(filters, 
                        kernel_size=(4,4), 
                        strides=(4,4),
                        kernel_initializer=kernel_init)(x)
        
    
    x = act_func(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model

#instantiate the U gen top initial blocks and print layer shapes
U_gen_top_init_layer_shape_lst = []
U_gen_top_init_fn = U_gen_top_init(leakyrelu, 512, kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42)) 
for layer in U_gen_top_init_fn.layers:
    U_gen_top_init_layer_shape_lst.append(layer.output_shape)
print(U_gen_top_init_layer_shape_lst)

#define random tensor set see so consistent and feed through the previous instantiated block
#print output tensor and shape
x_U_gen_top_init = tf.random.normal((1,4, 4, 512))
U_gen_top_init_output = U_gen_top_init_fn(x_U_gen_top_init)
print(U_gen_top_init_output[0])
print(U_gen_top_init_output[0].shape)

def U_gen_top_add(
    act_func,
    downsample_func,
    filters1,
    filters2, 
    image_shape,
    kernel_init=kernel_init,
):
    inputs = layers.Input(shape = (image_shape[0],image_shape[1],filters1))
    
    x = act_func(EqualizedConv2D(filters1,
                                kernel_size=(3,3), 
                                strides=(1,1),
                                kernel_initializer=kernel_init)(inputs))
    x = act_func(EqualizedConv2D(filters2, 
                                kernel_size=(3,3), 
                                strides=(1,1), 
                                kernel_initializer=kernel_init)(x))
    x = downsample_func(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model
#instantiate the U gen top additional blocks and print layer shapes
U_gen_top_add_layer_shape_lst = []
U_gen_top_add_fn = U_gen_top_add(leakyrelu, avgpooling2D, 512,512,(8,8,512), kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42)) 
for layer in U_gen_top_add_fn.layers:
    U_gen_top_add_layer_shape_lst.append(layer.output_shape)
print(U_gen_top_add_layer_shape_lst)

x_U_gen_top_add = tf.random.normal((1,8, 8, 512))
output_U_gen_top_add = U_gen_top_add_fn(x_U_gen_top_add)
print(output_U_gen_top_add[0])
print(output_U_gen_top_add[0].shape)


def U_connect(top, bottom, center=None, input_shape=(4,4), filters=512):
    
    inputs = layers.Input(shape = (input_shape[0],input_shape[1],filters))        
    
    if center == None:    
        x = top(inputs)
        x = bottom(x)

    else:
        h = top(inputs)
        x, _ = center(h)
        x = x+h
        x = bottom(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model
#unit test U_connect
top_G = U_gen_top_init_fn
bottom_G = U_gen_bottom_init_fn
center_G = None

u_conn_list_shape_lst=[]
U_connect_fn = U_connect(top_G, bottom_G, center_G, (4,4), 512)
for layer in U_connect_fn.layers:
    u_conn_list_shape_lst.append(layer.output_shape)
print(u_conn_list_shape_lst)

x_U_conn = tf.random.normal((1,4, 4, 512))
output_U_Conn = U_connect_fn(x_U_conn)
print(output_U_Conn[0])
print(output_U_Conn[0].shape)

def U_rgb_fadein(top,
                center,
                bottom,
                act_func,
                downsample_func,
                image_shape,
                filters1,
                filters2,
                kernel_init=kernel_init
                ):
    
    inputs = layers.Input(shape = (image_shape[0],image_shape[1], 3))
    alpha = layers.Input(shape = (1,))
    
    x = act_func(EqualizedConv2D(filters1, 
                                kernel_size=(1,1),        
                                strides=(1,1),
                                kernel_initializer=kernel_init)(inputs))
    x = top(x)
    if not center == None:
        h = downsample_func(inputs)
        
        h = act_func(EqualizedConv2D(filters2, 
                                    kernel_size=(1,1), 
                                    strides=(1,1),
                                    kernel_initializer=kernel_init)(h))

        fade_in = (1-alpha)*h+alpha*x
        w, _ = center(fade_in)
        x = x+w

    x,upsample = bottom(x)
    
    if x.shape[1] == 4 and upsample.shape[1] == 4:
        fade_in = EqualizedConv2D(3, 
                                    kernel_size=(1,1), 
                                    strides=(1,1),
                                    kernel_initializer=kernel_init)(x)
        
        fade_in = tf.math.tanh(fade_in)
    else:
        upsample = EqualizedConv2D(3, 
                                    kernel_size=(1,1), 
                                    strides=(1,1),
                                    kernel_initializer=kernel_init)(upsample)
        
        x = EqualizedConv2D(3, 
                            kernel_size=(1,1), 
                            strides=(1,1),
                            kernel_initializer=kernel_init)(x)
        
        fade_in = tf.math.tanh((1-alpha)*upsample+alpha*x)
    
    model = tf.keras.models.Model([inputs,alpha], fade_in)

    return model  

top_G = U_gen_top_init_fn
bottom_G = U_gen_bottom_init_fn
center_G = None

to_rgb_fadein_list_shape_lst=[]
U_rgb_fadein_fn = U_rgb_fadein(top_G, center_G, bottom_G,relu, avgpooling2D, (4,4), 512,512,kernel_init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42))
for layer in U_rgb_fadein_fn.layers:
    to_rgb_fadein_list_shape_lst.append(layer.output_shape)
print(to_rgb_fadein_list_shape_lst)

x_to_rgb_fadein = tf.random.normal((1,4, 4, 3))
alpha = tf.constant([[1.0]])
output_U_rgb_fadein = U_rgb_fadein_fn([x_to_rgb_fadein, alpha])
print(output_U_rgb_fadein)
print(output_U_rgb_fadein.shape)

###### disc ######

def final_block_disc(
    act_func,
    kernel_init=kernel_init,
    filters=512
):
    inputs = layers.Input(shape = (4,4,filters))
    
    x = minibatch_stddev_layer(inputs, group_size=4)
    
    x = EqualizedConv2D(filters, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        kernel_initializer=kernel_init)(x)
    x = act_func(x)
        
    x = EqualizedConv2D(filters, 
                        kernel_size=(4,4), 
                        strides=(4,4),
                        kernel_initializer=kernel_init)(x)
        
    
    x = act_func(x)
    x = EqualizedDense(1)(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model

#instantiate the final disc blocks and print layer shapes
final_block_disc_layer_shape_lst = []
final_block_disc_fn = final_block_disc(leakyrelu, tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42),512) 
for layer in final_block_disc_fn.layers:
    final_block_disc_layer_shape_lst.append(layer.output_shape)
print(final_block_disc_layer_shape_lst)

#define random tensor set see so consistent and feed through the previous instantiated block
#print output tensor and shape
x_final_block_disc = tf.random.normal((1,4, 4, 512))
final_block_disc_output = final_block_disc_fn(x_final_block_disc)
print(final_block_disc_output)
print(final_block_disc_output.shape)


def additional_block_disc(
    act_func,
    downsample_func,
    filters1,
    filters2, 
    image_shape,
    kernel_init=kernel_init,
):
    inputs = layers.Input(shape = (image_shape[0],image_shape[1],filters1))
    
    x = act_func(EqualizedConv2D(filters1,
                                kernel_size=(3,3),     
                                strides=(1,1),
                                kernel_initializer=kernel_init)(inputs))
    x = act_func(EqualizedConv2D(filters2, 
                                kernel_size=(3,3), 
                                strides=(1,1), 
                                kernel_initializer=kernel_init)(x))
    x = downsample_func(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model

#instantiate the additional discriminator blocks and print layer shapes
additional_block_disc_layer_shape_lst = []
additional_block_disc_fn = additional_block_disc(leakyrelu, avgpooling2D,512,512,(8,8),tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42)) 
for layer in additional_block_disc_fn.layers:
    additional_block_disc_layer_shape_lst.append(layer.output_shape)
print(additional_block_disc_layer_shape_lst)

#define random tensor set see so consistent and feed through the previous instantiated block
#print output tensor and shape
x_additional_block_disc = tf.random.normal((1,8, 8, 512))
additional_block_disc_output = additional_block_disc_fn(x_additional_block_disc)
print(additional_block_disc_output)
print(additional_block_disc_output.shape)

def connect_model(top, bottom, input_shape, filter2):
    
    inputs = layers.Input(shape = (input_shape[0],input_shape[1],filter2))
    
    x = top(inputs)
    x = bottom(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model

bottom = final_block_disc_fn
top = additional_block_disc_fn
conn_list_shape_lst=[]
connect_fn = connect_model(top, bottom, (8,8), 512)
for layer in connect_fn.layers:
    conn_list_shape_lst.append(layer.output_shape)
print(conn_list_shape_lst)

x_conn = tf.random.normal((1,8, 8, 512))
output_conn = connect_fn(x_conn)
print(output_conn)
print(output_conn.shape)


def from_rgb_fadein(top, bottom, 
                        act_func,
                        downsample_func,
                        image_shape,
                        filters1,
                        filters2,
                        kernel_init=kernel_init,
                        ):
    
    inputs = layers.Input(shape = (image_shape[0],image_shape[1], 3))
    alpha = layers.Input(shape = (1,))
    
    x = act_func(EqualizedConv2D(filters1, 
                                kernel_size=(1,1),        
                                strides=(1,1),
                                kernel_initializer=kernel_init)(inputs))
    x = top(x)
    if not bottom == None:
        h = downsample_func(inputs)
        
        h = act_func(EqualizedConv2D(filters2, 
                                    kernel_size=(1,1), 
                                    strides=(1,1),
                                    kernel_initializer=kernel_init)(h))

        fade_in = (1-alpha)*h+alpha*x
        # after from rgb
        x = bottom(fade_in)
    
    model = tf.keras.models.Model([inputs, alpha], x)
    return model  

top = final_block_disc(leakyrelu, tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42),512) 
bottom = None

from_rgb_fadein_shape_lst=[]
from_rgb_fadein_fn = from_rgb_fadein(top, bottom, leakyrelu,avgpooling2D,(4,4), 512,512,tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=42))
for layer in from_rgb_fadein_fn.layers:
    from_rgb_fadein_shape_lst.append(layer.output_shape)
print(from_rgb_fadein_shape_lst)

tf.random.set_seed(42)
x_from_rgb_fadein= tf.random.normal((1,4, 4, 3))
alpha = tf.constant([1.0])
output_from_rgb_fadein = from_rgb_fadein_fn([x_from_rgb_fadein,alpha])
print(output_from_rgb_fadein)
print(output_from_rgb_fadein.shape)