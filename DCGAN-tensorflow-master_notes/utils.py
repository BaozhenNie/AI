"""
Some codes from https://github.com/Newmu/dcgan_code
主要负责图像的一些基本操作，获取图像、保存图像、图像翻转，和利用moviepy模块可视化训练过程。
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
'''
pprint能够打印python的数据结构，看起来更直观。
PrettyPrinter类有三个参数：indent, depth, width

indent:展示数据时，缩进多少(每个递归层) 
depth:最多显示层级 
width:展示一行宽度，默认80
'''
pp = pprint.PrettyPrinter()
#猜测x:输入矩阵,k_h:filter高度, k_w:filter宽度
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

'''
用法：
import tensorflow as tf
import tensorflow.contrib.slim as slim

x1=tf.Variable(tf.constant(1,shape=[1],dtype=tf.float32),name='x11')
x2=tf.Variable(tf.constant(2,shape=[1],dtype=tf.float32),name='x22')
m=tf.train.ExponentialMovingAverage(0.99,5)
v=tf.trainable_variables()
for i in v:
    print("---")
        print(i)

print("Variables:\n")
slim.model_analyzer.analyze_vars(v,print_info=True)
print("Done")

运行结果：
---
<tf.Variable 'x11:0' shape=(1,) dtype=float32_ref>
---
<tf.Variable 'x22:0' shape=(1,) dtype=float32_ref>
Variables:

---------
Variables: name (type shape) [size]
---------
x11:0 (float32_ref 1) [1, bytes: 4]
x22:0 (float32_ref 1) [1, bytes: 4]
Total size of variables: 2
Total bytes of variables: 8
Done
'''
def show_all_variables():
　#tf.trainable_variable返回的是可训练的变量列表
  model_vars = tf.trainable_variables()
  #以下代码可打印出训练参数的变量信息
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)
#根据图像路径参数读取路径，根据灰度化参数选择是否进行灰度化。然后对图像参照输入的参数进行裁剪。
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)
#转换图片，并保存到指定位置
def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)
#判断grayscale参数是否进行范围灰度化，并进行类型转换为np.float
def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)
#转换图片
def merge_images(images, size):
  return inverse_transform(images)
#将images拼接成size[0]*size[1]的一个大image，images为一个batch(64)个image，size为[8 8]
def merge(images, size):
　#获得图片的高度h和宽度w
  h, w = images.shape[1], images.shape[2]
  #判断图片的通道数是不是３或４
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    #创建一个(size[0]*size[1]*单张图片大小)的数组,size[0]为纵排个数，size[1]为横排个数
    img = np.zeros((h * size[0], w * size[1], c))
    #遍历每张图片，idx:图片的索引
    for idx, image in enumerate(images):
      #i为要存放的图片所在行的第几个
      i = idx % size[1]
      #j为要存放在第几行
      j = idx // size[1]
      
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  #如果是灰度图  
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')
#我们这里的batch_size设置是64,merge的size设置为(8,8),将拼接好的图片存放在指定位置
def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)
#对图像进行裁剪
def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  #取出x的前两维值(即图像的高和宽)付给高，宽
  h, w = x.shape[:2]
  #将图像的h和w与crop的h，w相减除2，得到取整的值，即为图像要裁剪的起始位置
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  #将要裁剪下的部分resize
  #imresize is deprecated! imresize is deprecated in SciPy 1.0.0, and will be removed in 1.2.0. Use skimage.transform.resize instead. This function is used to resize the image
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])
#判断crop的值, 并对应resize返回对应的img
def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.
#推测是对图像反色，原理暂不清楚
def inverse_transform(images):
  return (images+1.)/2.
#未用到，暂时掠过
def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))
#利用moviepy.editor模块来制作动图，为了可视化用的。函数又定义了一个函数make frame(t)，首先根据图像集的长度和持续的时间做一个除法，然后返回每帧图像。最后视频修剪并制作成GIF动画
def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)
#定义visualize(sess, dcgan, config, option)函数。分为0、1、2、3、4种option。如果option=0，则之间显示生产的样本‘如果option=1，根据不同数据集不一样的处理，并利用前面的save_images()函数将sample保存下来；等等。本次在main.py中选用option=1。
def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

#定义image_manifold_size(num_images)函数。首先获取图像数量的开平方后向下取整的h和向上取整的w，然后设置一个assert断言，如果h*w与图像数量相等，则返回h和w，否则断言错误提示。
def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
