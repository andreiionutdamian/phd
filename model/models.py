# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:11:16 2020

@author: Andrei
"""

from model.layers import stem_block
from model.layers import inc_res_block
from model.layers import direct_convert_to_output_map
from model.layers import ds_res_block
from model.layers import shrink_block

from model import efficientnet 

import tensorflow as tf


def CloudifierNetV1(input_shape, 
                    inc_res_filters=[64, 128, 256], 
                    ds_res_filters=[256, 512, 1024],
                    n_classes=30):
  tf_input = tf.keras.layers.Input(input_shape, name='input')
  tf_x = tf_input
  tf_x = stem_block(tf_x)

  lst_out_maps_names = ['stem']
  lst_out_maps_generators = [tf_x]
  
  n_shrinks = 0
  
  for i,f in enumerate(inc_res_filters):
    name = 'ir_{}'.format(i+1)
    tf_x = inc_res_block(tf_x, n_filters=f, name=name)  
    
  lst_out_maps_generators.append(tf_x)
  lst_out_maps_names.append(name)
  
  for i,f in enumerate(ds_res_filters):
    name = 'dsr_{}'.format(i+1)
    if n_shrinks < 3:
      n_shrinks += 1
      tf_x = shrink_block(tf_x, name='shrink_'+str(n_shrinks))    
    tf_x = ds_res_block(tf_x, n_filters=f, name=name)
    lst_out_maps_generators.append(tf_x)
    lst_out_maps_names.append(name)

  lst_outmaps = direct_convert_to_output_map(lst_out_maps_generators, 
                                             output_shape=input_shape,
                                             input_names=lst_out_maps_names)    
  tf_x = tf.keras.layers.concatenate(lst_outmaps)
  
  tf_out = tf.keras.layers.Dense(units=n_classes, activation='softmax', name='readout')(tf_x)
  model = tf.keras.models.Model(tf_input, tf_out, name='CloudifierNetV1')
  return model

def IncResBlock(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = inc_res_block(tf_x, n_filters=256, name='inc_res_256')
  return tf.keras.models.Model(tf_inp, tf_x, name='02_IncResBlock')

def ShrinkDepthwiseSepRes(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = shrink_block(tf_x, name='ds_block_shrink')
  tf_x = ds_res_block(tf_x, n_filters=256, name='ds_res_256')
  return tf.keras.models.Model(tf_inp, tf_x, name='03_ShrinkDepthwiseSepRes')

def DepthwiseSepResBlock(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = ds_res_block(tf_x, n_filters=256, name='ds_res_256')
  return tf.keras.models.Model(tf_inp, tf_x, name='04_DepthwiseSepResBlock')


def StemBlock(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = stem_block(tf_x)
  return tf.keras.models.Model(tf_inp, tf_x, name='00_StemBlock')


def ShrinkBlock(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = shrink_block(tf_x)
  return tf.keras.models.Model(tf_inp, tf_x, name='01_ShrinkBlock')


def UpscaleBlock(input_shape):
  size = input_shape[-2]
  filters = [64, 256, 512, 1024]
  sizes = [size // (2**x) for x in range(2,6)]
  shapes = [(s,s,f) for s,f in zip(sizes, filters)]  
  inputs = [tf.keras.layers.Input(shapes[0], name='module_inp_1')] 
  inputs = inputs + [tf.keras.layers.Input(s,name='module_inp_'+str(i)) for i,s in enumerate(shapes)]
  tf_x = direct_convert_to_output_map(inputs, input_shape)
  return tf.keras.models.Model(inputs, tf_x, name='05_UpscaleBlock')
  
  
  


def EfficientBlock2(input_shape):  
  args = {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
          'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
  # drop_connect_rate: float, dropout rate at skip connections.          
  drop_connect_rate = 0.2
  blocks = 100
  b = 50 # middle
  drop_rate = drop_connect_rate * b / blocks 
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  for j in range(args.pop('repeats')):
    tf_x = efficientnet.block(tf_x, drop_rate=drop_rate, name='midl_eff_blk_{}'.format(
                                    j+1), 
                              **args)
  return tf.keras.models.Model(tf_inp, tf_x, name='EfficientBlock2')


if __name__ == '__main__':
  shape = (352, 352, 3)
  
  
  baselines = [
      EfficientBlock2(shape),
      efficientnet.EfficientNetB0(weights=None),
      efficientnet.EfficientNetB3(weights=None),
      efficientnet.EfficientNetB3(weights=None),
      efficientnet.EfficientNetB5(weights=None),
      efficientnet.EfficientNetB7(weights=None),
      tf.keras.applications.DenseNet121(weights=None),
      tf.keras.applications.Xception(weights=None),
      tf.keras.applications.MobileNetV2(weights=None)
  ]
  

  cloudifier_v1 = CloudifierNetV1(shape)
  
  cloudifier = [
      cloudifier_v1,
      StemBlock(shape),
      ShrinkBlock(shape),
      IncResBlock(shape),
      ShrinkDepthwiseSepRes(shape),
      DepthwiseSepResBlock(shape),
      UpscaleBlock(shape),
  ]
  
  names = [x.name for x in cloudifier] + ['base_' + x.name for x in baselines]
  models = cloudifier + baselines
  
  for model, name in zip(models, names):
    print("\n\n{}".format(name))
    #model.summary()
    
    tf.keras.utils.plot_model(model,to_file='img/'+name+'.png',
                              show_shapes=True,
                              show_layer_names=True)
    
