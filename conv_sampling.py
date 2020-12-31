  
# Copyright (C) 2020 Zach (Yuzhe) Ni 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
#
# Standard Library Imports
#


# downsampling for decoder
def downsample(filters, size, norm_layer=True):
    global init_func, gamma
    ret = keras.Sequential()
    ret.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=init_func, use_bias=False))
    if norm_layer:
        ret.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma))
    ret.add(layers.LeakyReLU())
    return ret
    
# upsampling for encoder
def upsample(filters, size, apply_dropout=False):
    global init_func, gamma
    ret = keras.Sequential(layers=[
                            layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=init_func,
                                      use_bias=False), 
                            tfa.layers.InstanceNormalization(gamma_initializer=gamma),
                            layers.ReLU()
    ])
    return result

# building encoder decoder model
def decoder_encoder_model(latent_dim):
    global init_func
    inputs = layers.Input(shape=[64,64,3])
    down  =[ ]
    up = []
    last = layers.Conv2DTranspose(3, 3,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=init_func,
                                  activation='tanh') 

    x = inputs
    inputs2 = layers.Input(shape=[latent_dim,])
    for d in down:
        x = d(x)
    flat = tf.keras.layers.Flatten()
    encode = tf.keras.layers.Dense(latent_dim + latent_dim)(flat(x))
    y = tf.keras.layers.Dense(units=_*_*_, activation=tf.nn.relu)(inputs2)
    y = tf.keras.layers.Reshape(target_shape=(_, _,_))(y)
    for u in up:
        y = u(y)

    decode = last(y)

    return keras.Model(inputs=inputs2, outputs=decode), keras.Model(inputs=inputs, outputs=encode)
