import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *
#定义了一堆变量：image_summary 、scalar_summary、histogram_summary、merge_summary、SummaryWriter，都是从相应的tensorflow中获取的。如果可是直接获取，则获取，否则从tf.summary中获取。
try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter
#用来拼接多个tensor
'''
利用dir(tf)判断”concat_v2”是否在里面，如果在的话，定义一个concat(tensors, axis, *args, **kwargs)函数，并返回tf.concat_v2(tensors, axis, *args, **kwargs)；否则也定义concat(tensors, axis, *args, **kwargs)函数，只不过返回的是tf.concat(tensors, axis, *args, **kwargs)。其中，tf.concat使用如下：

t1=tf.constant([[1,2,3],[4,5,6]])
t2=tf.constant([[7,8,9],[10,11,12]])
t3=tf.concat([t1,t2],0)
t4=tf.concat([t1,t2],1)
print t1
print t2
print t3
print t4
'''
if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)
#定义一个batch_norm类，包含两个函数init和call函数。首先在init(self, epsilon=1e-5, momentum = 0.9, name=”batch_norm”)函数中，定义一个name参数名字的变量，初始化self变量epsilon、momentum 、name。在call(self, x, train=True)函数中，利用tf.contrib.layers.batch_norm函数批处理规范化。
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
#连接x,y与Int32型的[x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]]维度的张量乘积。
def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
#定义conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2,d_w=2, stddev=0.02,name=”conv2d”)函数。卷积函数：获取随机正态分布权值、实现卷积、获取初始偏置值，获取添加偏置值后的卷积变量并返回。
def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"): 
    #定义conv2节点：随机初始化W，
    # filter: [height, width, output_channels, in_channels]
    #W的shape为[5       5        input[-1]      output_dim]
    #卷积在每一维的步长[1 2 2 1]
    #padding='SAME'时：output size(h,w) = input(h,w)/stride = input(h,w)/2
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    #biase 舒适化为 0
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
#反卷积函数：获取随机正态分布权值、解卷积，获取初始偏置值，获取添加偏置值后的卷积变量，判断with_w是否为真，真则返回解卷积、权值、偏置值，否则返回解卷积。
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    #初始化W W shape:[5   5    output_shape   input size[-1]]注意：此处output size与input size与conv2是不同的，原因应该与tf的conv2实现内部实现有关系，暂时不清楚
    # W b 都是可以共享的，之后会与sampler共享
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    #因为版本问题尝试使用conv2d_transpose，出现错误，使用deconv2d实现反卷积
    #反卷积与卷积的filter矩阵正好是转至关系
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])
    #偏差使用0初始化
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    
    if with_w:
      return deconv, w, biases
    else:
      return deconv
#leaky relu函数定义     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
#进行线性运算，获取一个随机正态分布矩阵，获取初始偏置值，如果with_w为真，则返回xw+b，权值w和偏置值b；否则返回xw+b。
#第一项参数为input，第二项参数为output size
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  #获取input size
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    #根据input size 和output size，以及stddev正态初始化矩阵matrix即W
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    #根据output size，初始化biase，初始化的值由linear的参数bias_start决定
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    #由参数with_w来判定是否要返回w及b
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
