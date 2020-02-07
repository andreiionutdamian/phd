# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:11:16 2020

@author: Andrei
"""

from model.layers import stem_block, inc_res_block, convert_to_output_map, ds_res_block, shrink_block

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

  lst_outmaps = convert_to_output_map(lst_out_maps_generators, 
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
  return tf.keras.models.Model(tf_inp, tf_x, name='IncResBlock')

def DSResBlockS(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = shrink_block(tf_x, name='ds_block_shrink')
  tf_x = ds_res_block(tf_x, n_filters=256, name='ds_res_256')
  return tf.keras.models.Model(tf_inp, tf_x, name='DSResBlockS')

def DSResBlock(input_shape):
  tf_inp = tf.keras.layers.Input(input_shape,name='module_input')
  tf_x = tf_inp
  tf_x = ds_res_block(tf_x, n_filters=256, name='ds_res_256')
  return tf.keras.models.Model(tf_inp, tf_x, name='DSResBlock')


if __name__ == '__main__':
  shape = (352, 352, 3)
  cloudifier_v1 = CloudifierNetV1(shape)
  irblock = IncResBlock(shape)
  dsblock1 = DSResBlock(shape)
  dsblock2 = DSResBlockS(shape)
  
  models = [cloudifier_v1, irblock, dsblock1, dsblock2]
  
  for model in models:
    print("\n\n{}".format(model.name))
    model.summary()
    
    tf.keras.utils.plot_model(model,to_file='img/'+model.name+'.png',
                              show_shapes=True,
                              show_layer_names=True)
    
