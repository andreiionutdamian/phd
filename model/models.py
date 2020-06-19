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


class GatingUnit(tf.keras.layers.Layer):
  def __init__(self, 
               layer,
               activation='sigmoid',
               **kwargs):
    self.gate_trans = layer
    self.gate_activ = tf.keras.activations.get(activation)
    
    super().__init__(**kwargs)
    
  def call(self, inputs):
    tf_source = inputs[0]
    tf_value1 = inputs[1]
    tf_value2 = inputs[2]
    
    tf_gate = self.gate_activ(self.gate_trans(tf_source))
    tf_out = tf_gate * tf_value1 + (1 - tf_gate) * tf_value2
    return tf_out
  
  def get_config(self):
    config = {
        'layer' : tf.keras.layers.serialize(self.gate_trans),
        'activation' : tf.keras.activations.serialize(self.gate_activ)
        }
    base_config = super().get_config()
    cfg =  dict(list(base_config.items()) + list(config.items()))
    return cfg
  
class MultiGatedUnit(tf.keras.layers.Layer):
  def __init__(self,
               layer,
               gates_initializer='glorot_uniform',
               bypass_initializer='glorot_uniform',
               gating_activation='sigmoid',
               **kwargs,
               ):
    self.gates_initializer = gates_initializer
    self.bypass_initializer = bypass_initializer
    self.gating_activation = gating_activation
    self.layer_class = layer.__class__
    
    
    base_layer_config = layer.get_config()
    name = base_layer_config.pop('name') if 'name' in base_layer_config else ''
    if name == '':
      name = kwargs.get('name') if 'name' in kwargs else 'MGU'
    
    self._name = name
      
    self.activation = kwargs.get('activation') if 'activation' in kwargs else base_layer_config.get('activation')
    if base_layer_config.get('activation') not in [None, 'linear']:
      base_layer_config['activation'] = 'linear'
      
    if self.activation == 'linear':
      raise ValueError('Cannot have MGU with linear activation. Either set host MGU or the layer activation')

    if 'activation' in kwargs:
      kwargs.pop('activation')
    
    self.base_layer_config = base_layer_config
    
    self.layer = layer.__class__.from_config(self.base_layer_config)
    assert self.layer.activation.__name__ == 'linear'
    self.layer_activation = tf.keras.activations.get(self.activation)
    
    # bypass
    self.bypass = self._create_layer('bypass', self.bypass_initializer, use_bias=False)
    
    self.bn_pre = tf.keras.layers.BatchNormalization()
    self.bn_pos = tf.keras.layers.BatchNormalization()    
    self.ln_pos = tf.keras.layers.LayerNormalization()
    
    self.g_bpre_bpos = GatingUnit(
        layer=self._create_layer('g_bpre_bpos', self.gates_initializer, use_bias=True),
        activation=self.gating_activation,
        name=name+'_gu1')
    
    self.g_bn_ln = GatingUnit(
        layer=self._create_layer('g_bn_ln', self.gates_initializer, use_bias=True),
        activation=self.gating_activation,
        name=name+'_gu2')
    
    self.g_norm_non = GatingUnit(
        layer=self._create_layer('g_norm_non', self.gates_initializer, use_bias=True),
        activation=self.gating_activation,
        name=name+'_gu3')
    
    self.g_proc_skip = GatingUnit(
        layer=self._create_layer('g_proc_skip', self.gates_initializer, use_bias=True),
        activation=self.gating_activation,
        name=name+'_gu4')
    
    super().__init__(**kwargs)
    
  def call(self, inputs):
    tf_bypass = self.bypass(inputs)
    tf_x = self.layer(inputs)
    tf_x_act = self.layer_activation(tf_x)
    
    tf_x_bn_act = self.layer_activation(self.bn_pre(tf_x))
    tf_x_act_bn = self.bn_pos(tf_x_act)
    tf_x_act_ln = self.ln_pos(tf_x_act)
    
    tf_bpre_bpos = self.g_bpre_bpos([inputs, tf_x_bn_act, tf_x_act_bn])
    tf_bn_ln = self.g_bn_ln([inputs, tf_bpre_bpos, tf_x_act_ln])
    tf_norm_non = self.g_norm_non([inputs, tf_bn_ln, tf_x_act])
    
    tf_proc_noproc = self.g_proc_skip([inputs, tf_norm_non, tf_bypass])
    
    tf_out = tf_proc_noproc
    return tf_out
    
            
  def _create_layer(self, name, kernel_initializer, use_bias):
    layer_config = self.base_layer_config.copy()
    layer_config['activation'] = 'linear'
    layer_config['name'] = self._name + '_' + name
    layer_config['kernel_initializer'] = tf.keras.initializers.get(kernel_initializer)
    return self.layer_class.from_config(layer_config)
    

  def get_config(self):
    config = {
        'layer' :  tf.keras.layers.serialize(self.layer),
        'activation' : self.activation,
        'gating_activation': tf.keras.activations.serialize(self.gating_activation),
        'gates_initializer': tf.keras.initializers.serialize(self.gates_initializer),
        'bypass_initializer': tf.keras.initializers.serialize(self.bypass_initializer),
        }
    base_config = super().get_config()
    cfg =  dict(list(base_config.items()) + list(config.items()))
    return cfg
  


def CloudifierNetV0(input_shape, 
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
  model = tf.keras.models.Model(tf_input, tf_out, name='CloudifierNetV0')
  return model

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
    tf_x = MultiGatedUnit(
        layer=tf.keras.layers.SeparableConv2D(f, 3, padding='same', name=name),
        activation='relu'
        )(tf_x)
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
  filters = [128, 256, 384] #, 512]
  sizes = [size // (2**x) for x in range(2,2 + len(filters))]
  shapes = [(s,s,f) for s,f in zip(sizes, filters)]  
  inputs = [tf.keras.layers.Input(s,name='level_{}_input'.format(i+1)) for i,s in enumerate(shapes)]
  lst_outs = direct_convert_to_output_map(inputs, input_shape)
  tf_x = tf.keras.layers.concatenate(lst_outs, name='concatenated_upscaled_volumes')
  return tf.keras.models.Model(inputs, tf_x, name='05_UpscaleBlock')
  
  
  


def EfficientBlock(input_shape):  
  args = {'kernel_size': 5, 'repeats': 3, 'filters_in': 100, 'filters_out': 100,
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


def MGUSC(input_shape):
  layer = tf.keras.layers.SeparableConv2D(128, 3)

  def _create_layer(name):
    base_layer_config = layer.get_config()
    layer_config = base_layer_config.copy()
    layer_config['activation'] = 'linear'
    layer_config['name'] = name    
    return layer.__class__.from_config(layer_config)   
  
  layer_activation_pre_bn = tf.keras.layers.Activation('relu', name='ReLU_pre_bn')
  layer_activation_post_bn = tf.keras.layers.Activation('relu', name='ReLU_post_bn')
  bypass = _create_layer('bypass')
  bn_pre = tf.keras.layers.BatchNormalization()
  bn_pos = tf.keras.layers.BatchNormalization()    
  ln_pos = tf.keras.layers.LayerNormalization()
  
  g_bpre_bpos = GatingUnit(
        layer=_create_layer('l_bpre_bpos'),
        activation='sigmoid',
        name='gate_bpre_bpos'
        )
  
  g_bn_ln = GatingUnit(
        layer=_create_layer('l_bn_ln'),
        activation='sigmoid',
        name='gate_bn_ln'
        )
    
  g_norm_non = GatingUnit(
        layer=_create_layer('l_norm_non'),
        activation='sigmoid',
        name='gate_norm_non'
        )
    
  g_proc_skip = GatingUnit(
        layer=_create_layer('l_proc_skip'),
        activation='sigmoid',
        name='gate_proc_skip'
        )
  
  inputs = tf.keras.layers.Input(input_shape, name='input')
  tf_bypass = bypass(inputs)
  tf_x = layer(inputs)
  tf_x_act = layer_activation_pre_bn(tf_x)
  
  tf_x_bn_act = layer_activation_post_bn(bn_pre(tf_x))
  tf_x_act_bn = bn_pos(tf_x_act)
  tf_x_act_ln = ln_pos(tf_x_act)
  
  tf_bpre_bpos = g_bpre_bpos([inputs, tf_x_bn_act, tf_x_act_bn])
  tf_bn_ln = g_bn_ln([inputs, tf_bpre_bpos, tf_x_act_ln])
  tf_norm_non = g_norm_non([inputs, tf_bn_ln, tf_x_act])
  
  tf_proc_noproc = g_proc_skip([inputs, tf_norm_non, tf_bypass])
  
  tf_out = tf.keras.layers.Lambda(function=lambda x:x, name='Ouput')(tf_proc_noproc)
  return tf.keras.models.Model(inputs, tf_out, name='06_MGU_SC')  


if __name__ == '__main__':
  shape = (352, 352, 3)
  
  
  baselines = [
#      EfficientBlock(shape),
#      efficientnet.EfficientNetB0(weights=None),
#      efficientnet.EfficientNetB3(weights=None),
#      efficientnet.EfficientNetB3(weights=None),
#      efficientnet.EfficientNetB5(weights=None),
#      efficientnet.EfficientNetB7(weights=None),
#      tf.keras.applications.DenseNet121(weights=None),
#      tf.keras.applications.Xception(weights=None),
#      tf.keras.applications.MobileNetV2(weights=None)
  ]
  

  cloudifier_v0 = CloudifierNetV0(shape)
  cloudifier_v1 = CloudifierNetV1(shape)
  
  cloudifier = [
      # UpscaleBlock(shape),
      # cloudifier_v0,
      # cloudifier_v1,
      # StemBlock(shape),
      # ShrinkBlock(shape),
      # IncResBlock(shape),
      # ShrinkDepthwiseSepRes(shape),
      # DepthwiseSepResBlock(shape),
      MGUSC(shape)
  ]
  
  names = [x.name for x in cloudifier] + ['base_' + x.name for x in baselines]
  models = cloudifier + baselines
  
  for model, name in zip(models, names):
    print("\n\n{}".format(name))
    #model.summary()
    
    tf.keras.utils.plot_model(model,to_file='img/'+name+'.png',
                              show_shapes=True,
                              show_layer_names=True)
    
