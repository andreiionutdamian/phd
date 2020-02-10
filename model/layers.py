# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:56:58 2020

@author: Andrei
"""

import tensorflow as tf

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
  
    x = tf.keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=use_bias,
                               name=name+'_c_k{}'.format(kernel_size)
                               )(x)
    if not use_bias:
        bn_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = tf.keras.layers.Activation(activation, name=ac_name)(x)
    return x

def stem_block(tf_input):
  tf_x = tf_input
  tf_x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(tf_x)
  tf_x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, 
                                use_bias=False, name='stem_c1')(tf_x)
  tf_x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='stem_bn1')(tf_x)
  tf_x = tf.keras.layers.Activation('relu', name='stem_relu1')(tf_x)


  tf_x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(tf_x)
  tf_x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, 
                                use_bias=False, name='stem_c2')(tf_x)
  tf_x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='stem_bn2')(tf_x)
  tf_x = tf.keras.layers.Activation('relu', name='stem_relu2')(tf_x)
  return tf_x

def shrink_block(tf_input, name='shrink', stride=2, kernel=3):
  tf_x = tf_input
  n_in_filters = tf.keras.backend.int_shape(tf_input)[-1]
  tf_x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)),
                                       name=name+'_zpad')(tf_x)
  tf_x = tf.keras.layers.Conv2D(filters=n_in_filters, kernel_size=kernel,
                                strides=stride,
                                padding='valid',
                                use_bias=False, activation=None,
                                name=name+'_cnv_k{}_s{}'.format(kernel, stride)
                                )(tf_x)
  return tf_x
  
  
  
def inc_res_block(tf_input, n_filters, activation='relu', scale=1, name=''):
  # we assume channel last
  n_in_filters = tf.keras.backend.int_shape(tf_input)[-1]
  tf_b1 = conv2d_bn(tf_input, filters=n_filters, kernel_size=1, name=name+'_b1')
  
  tf_b2 = conv2d_bn(tf_input, filters=n_filters, kernel_size=1, name=name+'_b21')
  tf_b2 = conv2d_bn(tf_b2, filters=n_filters, kernel_size=3, name=name+'_b22')
  
  tf_b3 = conv2d_bn(tf_input, filters=n_filters, kernel_size=1, name=name+'_b31')
  tf_b3 = conv2d_bn(tf_b3, filters=n_filters, kernel_size=5, name=name+'_b32')
  
  tf_mixed = tf.keras.layers.concatenate([tf_b1, tf_b2, tf_b3], name=name+'_mix')
  
  tf_mixed_reduced = conv2d_bn(tf_mixed, filters=n_in_filters,
                               kernel_size=1, activation=None,
                               use_bias=True, # no BN as a result
                               name=name+'_mixreshape')
  if scale != 1:
    tf_x = scaled_residual(tf_x=tf_mixed_reduced, 
                           tf_input=tf_input, 
                           scale=scale, 
                           name=name+'_res_scal')
  else:
    tf_x = tf.keras.layers.add([tf_input, tf_mixed_reduced], name=name+'_res')
    
  if activation is not None:
    tf_x = tf.keras.layers.Activation(activation, name=name+'_blk_'+activation)(tf_x)
  return tf_x
  
  

def ds_res_block(tf_input, n_filters, n_layers=3, name=''):
  tf_x = tf_input
  n_in_filters = tf.keras.backend.int_shape(tf_input)[-1] 
  if n_in_filters != n_filters:
    tf_residual = tf.keras.layers.Conv2D(filters=n_filters,
                                         kernel_size=1,
                                         padding='same',
                                         name=name+'_resid_c2d_k1',
                                         use_bias=False,
                                         )(tf_input)
  else:
    tf_residual = tf_input
  for i in range(1, n_layers+1):
    tf_x = tf.keras.layers.SeparableConv2D(n_filters, 3,
                                           use_bias=False,
                                           padding='same',
                                           name=name+'_sc{}'.format(i))(tf_x)
    tf_x = tf.keras.layers.BatchNormalization(name=name+'_bn{}'.format(i))(tf_x)
    tf_x = tf.keras.layers.Activation('relu', name=name + '_relu{}'.format(i))(tf_x)
  tf_x = tf.keras.layers.add([tf_x, tf_residual], name=name+'_resid')
  return tf_x


def convert_to_output_map(lst_inputs, output_shape, input_names=[]):
  lst_outputs = []
  for i, tf_input in enumerate(lst_inputs):
    input_shape = tf.keras.backend.int_shape(tf_input)
    name = input_names[i] if len(input_names) > 0 else 'tra_{}'.format(i+1)
    input_H = input_shape[-2]
    output_H = output_shape[0]
    stride = output_H // input_H
    f = input_shape[-1]
    tf_out = tf.keras.layers.Conv2DTranspose(filters=f,
                                             kernel_size=3,
                                             strides=stride,
                                             padding='valid',
                                             name=name+'_trns_s'+str(stride))(tf_input)
    lst_outputs.append(tf_out)
  return lst_outputs


def scaled_residual(tf_x, tf_input, scale, name=''):
  tf_out = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                  output_shape=tf.keras.backend.int_shape(tf_input)[1:],
                                  arguments={'scale': scale},
                                  name=name)([tf_input, tf_x])
  return tf_out


  
  
  
  

  