from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
#计算卷积后的输出大小
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))
#DCGAN
class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]，为样本的种类即labels个数，如mnist的y_dim=10
      z_dim: (optional) Dimension of dim for Z. [100]，z is the input size for generator
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    定义类的初始化函数 init。主要是对一些默认的参数进行初始化。包括session、crop、批处理大小batch_size、样本数量sample_num、输入与输出的高和宽、各种维度、生成器与判别器的批处理、数据集名字、灰度值、构建模型函数，需要注意的是，要判断数据集的名字是否是mnist，是的话则直接用load_mnist()函数加载数据，否则需要从本地data文件夹中读取数据，并将图像读取为灰度图。
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

#定义每一层的batch normalization 对象
#y_dim为被指定时，G，D比指定时多一个batch normalization环节
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
#指定时mnist数据集，则load_mnist数据集
    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    else:
        #返回./data/dataset_name/.jpg格式的所有图片的路径，即返回所有数据图片的路径，并通过imread读取
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()
'''
定义构建模型函数build_model(self)。
首先判断y_dim，然后用tf.placeholder占位符定义并初始化y。
判断crop是否为真，是的话是进行测试，图像维度是输出图像的维度；否则是输入图像的维度。
利用tf.placeholder定义inputs，是真实数据的向量。
定义并初始化生成器用到的噪音z，z_sum。
再次判断y_dim，如果为真，用噪音z和标签y初始化生成器G、用输入inputs初始化判别器D和D_logits、样本、用G和y初始化D_和D_logits；如果为假，跟上面一样初始化各种变量，只不过都没有标签y。
将5中的D、D_、G分别放在d_sum、d__sum、G_sum。
定义sigmoid交叉熵损失函数sigmoid_cross_entropy_with_logits(x, y)。都是调用tf.nn.sigmoid_cross_entropy_with_logits函数，只不过一个是训练，y是标签，一个是测试，y是目标。
定义各种损失值。真实数据的判别损失值d_loss_real、虚假数据的判别损失值d_loss_fake、生成器损失值g_loss、判别器损失值d_loss。
定义训练的所有变量t_vars。
定义生成和判别的参数集。
最后是保存。
'''

#tf.histogram（对应tensorboard中的scalar）和tf.scalar函数（对应tensorboard中的distribution和histogram）是制作变化图表的，，两者差不多，一般是第一项字符命名，第二项就是要记录的变量了，
#最后用tf.summary.merge_all对所有训练图进行合并打包，最后必须用sess.run一下打包的图，并添加相应的记录。
  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None
#如果crop为true，说明图像需要裁剪，则image_dims为output size
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)
#D表示真实数据的判别器输出，D_表示生成数据的判别器输出
    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
#对于真实数据，判别器的损失函数d_loss_real为判别器输出与1的交叉熵
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
#对于生成数据，判别器的损失函数d_loss_fake为输出与0的交叉熵    
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
#生成器的损失函数是g_loss判别器对于生成数据的输出与1的交叉熵。
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
#判别器的损失函数d_loss=d_loss_real+d_loss_fake                         
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
#用于获得可训练的变量
    t_vars = tf.trainable_variables()
#将判别器与生成器的可训练变量分开
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
#用于模型的保存
    self.saver = tf.train.Saver()

  def train(self, config):
#选用Adam下降方法设置判别器和生成器的优化器
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
#兼容不同的tensorflow版本
#初始化所有变量
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
#打包相关的训练图，将G与D的相关结果分开打包，用于分开训练，打包的成员见如下代码
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
#定义writer,主要是用来定义监控结果的输出目录
    self.writer = SummaryWriter("./logs", self.sess.graph)
'''
以上部分为优化器以及SummaryWriter的定义
以下开始为变量赋值
'''
#初始化噪声输入z，在[-1,1]之间随机取值(服从均匀分布)，z的shape为[self.sample_num , self.z_dim]
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
#预处理得到输入image
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
          #如果grayscale为1，直接把最后一维channels舍掉
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    #load模型
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
#train epoch次，batch
    for epoch in xrange(config.epoch):
'''
train 与上边samples的输入图片的提取相类似，不同的是训练时要将输入图片分成多个batch
'''
      if config.dataset == 'mnist':
          #如下得到将数据集分为多少个batch
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        self.data = glob(os.path.join(
          "./data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)
#训练噪声z的舒适化，服从[-1 1]上的均匀分布
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          #对D进行一次优化，并监控D的loss值
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
    #add_summary用于给对应的summary collection成员添加新的值，在下一次sess.run[summary_str]时写到对应的监控里 
#每优化一次添加一次
          self.writer.add_summary(summary_str, counter)

          # Update G network
          #对G进行一次优化
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          #对G进行两次训练，避免D的loss为0，导致很难训练
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z, self.y:batch_labels })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))
#np.mod(a,n)相当于a%n, 每优化一百次 保存一次结果图像
        if np.mod(counter, 100) == 1:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")
#每500次保存一次模型
        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
#判断是否样本种类即样本labels，即判断y_dim的值，如mnist的y_dim=10，若是y_dim有赋值，即数据labels存在，则在卷积中拼接y(条件信息),作为网络层的输入，若没有赋值y_dim，则类似生成动漫头像一类的数据集
      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
#总结：无pooling层，四层卷积，除第一层未使用batch normaliazation外，其他层均使用，第四层的输出reshape为[self.batch_size, -1]，再输入给linear，最后使用sigmoid
        return tf.nn.sigmoid(h4), h4
'''
总结：y_dim有值时
(image + yb <- y)
       |
       V
       x - input
       |
       V
convolution -> lrelu
            |
            V
           (h0 + yb)
            |
            V
       convolution -> batch normalization -> lrelu
                            |
                            V
                           (h1 -> reshape -> [batch_size,-1] + y)
                                          |
                                          V
                                       linear -> batch normalization -> lrelu
                                                         |
                                                         V
                                                        (h2 + y)
                                                            |
                                                            V
                                                          linear
                                                            |
                                                            V
                                                            h3
                                                            |
                                                            V
                                                         sigmoid
    三层卷积层，两层线性层，输出为sigmoid                                         

'''
      else:
        #将y reshape成[self.batch_size, 1, 1, self.y_dim]
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        #image size is [self.batch_size, img_height, img_width, channels], 将image与yb拼接成[self.batch_size, img_height, img_width, channels + self.y_dim]矩阵，共同作为卷积层的输入
        #相当于使用conditional GAN，在输入图像中加入标签作为条件信息
        x = conv_cond_concat(image, yb)
        #将x做卷积，并使用leaky relu函数
#layer one：
        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        #将第一层的输出值h0，与yb拼接
        h0 = conv_cond_concat(h0, yb)
#layer two：
        #进行第二层卷积，在激励函数之前，加入batch normalization
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        #将输出reshape成[self.batch_size, X],-1表示系统自动计算另一维的数量
        #以下使用线性单元
        h1 = tf.reshape(h1, [self.batch_size, -1])
        #将第二层的输出与y拼接，因h1被reshape成[self.batch_size, -1}，因此与y[self.batch_size, self.y_dim]进行拼接，加入条件信息
        h1 = concat([h1, y], 1)
        #第三层使用线性单元，并在leaky relu之前加入batch normalization
#layer three：
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        #继续拼接y作为条件信息
        h2 = concat([h2, y], 1)
#layer four:
        #再次执行线性单元
        h3 = linear(h2, 1, 'd_h3_lin')
        #因为判别器输出是两分类，因此sigmoid作为输出函数
        return tf.nn.sigmoid(h3), h3
'''
定义生成器函数generator(self, z, y=None)。

利用with tf.variable_scope(“generator”) as scope，在一个作用域 scope 内共享一些变量。
根据y_dim是否为真，进行判别网络的设置。
如果为假：首先获取输出的宽和高，然后根据这一值得到更多不同大小的高和宽的对。然后获取h0层的噪音z，权值w，偏置值b，然后利用relu激励函数。h1层，首先对h0层反卷积得到本层的权值和偏置值，然后利用relu激励函数。h2、h3等同于h1。h4层，反卷积h3，然后直接返回使用tanh激励函数后的h4。
如果为真：首先也是获取输出的高和宽，根据这一值得到更多不同大小的高和宽的对。然后获取yb和噪音z。h0层，使用relu激励函数，并与1连接。h1层，对线性全连接后使用relu激励函数，并与yb连接。h2层，对反卷积后使用relu激励函数，并与yb连接。最后返回反卷积、sigmoid处理后的h2。
'''
  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        #如同discriminator，先判断样本的labels(y_dim)是否赋值，没有赋值则不拼接labels(y或yb)作为条件信息，作为下一层的输入
      if not self.y_dim:
        #conv_out_size_same第一项为输入参数的shape，第二项为stride，返回反卷积后的tensor shape
        #以下计算出反卷积的每一层的高，宽，即output size，从而做deconv2的输入参数
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        #z为随机生成的样本(符合均匀分布)通过linear单元，输出的tensor shape：[batch_size gf_dim*8*s_h16*s_w16]
#layer one: linear + reshape + batch normalization + relu -> tensor shape:[batch_size, s_h16, s_w16, self.gf_dim * 8]
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
        #对z进行reshape得到tensor shape：[batch_size, s_h16, s_w16, self.gf_dim * 8],并进行batch normalization + relu
        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))
#layer two: deconv2d + batch normalization + relu -> tensor shape: [batch_size, s_h8, s_w8, self.gf_dim*4]
        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))
#layer three: deconv2d + batch normalization + relu -> tensor shape: [batch_size, s_h4, s_w4, self.gf_dim*4]
        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))
#layer four: deconv2d + batch normalization + relu -> tensor shape: [batch_size, s_h2, s_w2, self.gf_dim*4]
        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))
#layer five: deconv2d + tanh -> [batch_size, s_h, s_w, self.c_dim]
        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:#y_dim有值，可以用来生成mnist图像
          #output_height,output_width分别为输出图像的高，宽
          #如mnist数据集输入则为s_h=28, s_w=28
          #从而得s_h2=s_w2=14,s_h4=s_w4=7
          #此处使用的z服从平均分布的随机分布
        s_h, s_w = self.output_height, self.output_width
        #每经过一层卷积，输出tensor shape 扩大一倍
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)
'''
总结：
第一层：linear + batch normalization + relu + 拼接y : 输出的tensor shape:[batch_size, gfc_dim+y_dim]
第二层：linear + batch normalization + relu + reshape + 拼接yb ： 输出的tensor shape:[batch_size, s_h4, s_w4, self.gf_dim * 2 + y_dim]
第三层：deconv2 + batch normalization + relu + 拼接yb : 输出的tensor shape: [batch_size, s_h2, s_w2, self.gf_dim * 2 + y_dim]
第四层：deconv2 + sigmoid : 输出的tensor shape : [self.batch_size, s_h, s_w, self.c_dim]
'''
        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        #拼接labels作为输入的条件信息
        #z的维度[64 100],yb的维度是[64 1 1 10]
        #z拼接y之后得到一个维度为[64 110]的tensor
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)
#layer one：
        #第一层为线性层，输出tensor的维度[63, gfc_dim],并加入batch normalization再输给relu，对输出再拼接y，作为下一层的输入
        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)
#layer two：
        #第二层继续使用线性，batch_norm及relu，输出的tensor维度为[batch_size, gf_dim*2*s_h4*s_w4]
        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        #对输出reshape成[batch_size, s_h4, s_w4, self.gf_dim * 2]的图片叠加格式
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        #拼接上yb作为下一层输入
        h1 = conv_cond_concat(h1, yb)
#layer three：
        #第三层使用反卷积使输出扩大，输出的tensor size为[batch_size, s_h2, s_w2, self.gf_dim * 2]
        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)
        #拼接yb作为条件信息，再进行一层反卷积最后输入给sigmoid
#layer four：
        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
'''
定义sampler(self, z, y=None)函数。

利用tf.variable_scope(“generator”) as scope，在一个作用域 scope 内共享一些变量。
对scope利用reuse_variables()进行重利用。
根据y_dim是否为真，进行判别网络的设置。
然后就跟生成器差不多，不在赘述。
生成器的W及b与generator中的共享
'''
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
#主要是针对mnist数据集设置的，所以暂且不考虑，过
  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec
#返回数据集名字，batch大小，输出的高和宽。
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
#保存训练好的模型。创建检查点文件夹，如果路径不存在，则创建；然后将其保存在这个文件夹下。
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
#恢复保存的模型，获取路径，重新存储检查点，并且计数。打印成功读取的提示；如果没有路径，则打印失败的提示。
  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
