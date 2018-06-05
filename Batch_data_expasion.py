import skimage
from skimage import io
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageChops
import cv2
#root_path为图像根目录，img_name为图像名字

import pprint
#import this to use flags
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("images_dir", "./images", "Directory name saving the origin images need to preprocessing.")
flags.DEFINE_string("img_save_path", "./images_by_processing", "Directory name to save the processing images.")
flags.DEFINE_multi_integer("translation", None, "Translate image.")
flags.DEFINE_bool("flip", False, "Flip image.")
flags.DEFINE_bool("contrast", False, "Contrast image.")
flags.DEFINE_integer("rotation", None, "Rotation image.")
flags.DEFINE_bool("G_noise", False, "Add Gaussian noise followed the Standard normal distribution into image.")
flags.DEFINE_bool("Color_dithering", False, "Add color dithering into image.")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()

def translation(img_path, img_name, off = 0): #平移
    img = Image.open(img_path)
    if len(FLAGS.translation) == 2:
        img_offset = ImageChops.offset(img, FLAGS.translation[0], FLAGS.translation[1])
    elif len(FLAGS.translation) == 2:
        img_offset = ImageChops.offset(img, FLAGS.translation[0], 0)
    else:
        print('flags --translation should not be specify more than three times.')

    img_offset.save(os.path.join(FLAGS.img_save_path, img_name + '_offset.jpg'))
    #img_offset.save(os.path.join(img_path.split('.')[0] + '_offset.jpg'))
    #return offset

def flip(img_path, img_name):   #翻转图像
    img = Image.open(img_path)
    flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flip_img.save(os.path.join(FLAGS.img_save_path, img_name + '_flip.jpg'))
    #flip_img.save(os.path.join(img_name.split('.')[0] + '_flip.jpg'))
    #return flip_img

def aj_contrast(img_path, img_name): #调整对比度 两种方式 gamma/log
    image = io.imread(img_path)
    gam= skimage.exposure.adjust_gamma(image, 0.5)
    io.imsave(os.path.join(FLAGS.img_save_path, img_name + '_gam.jpg'),gam)
    #skimage.io.imsave(os.path.join(img_path.split('.')[0] + '_gam.jpg'),gam)
    log= skimage.exposure.adjust_log(image)
    io.imsave(os.path.join(FLAGS.img_save_path, img_name + '_log.jpg'),gam)
    #skimage.io.imsave(os.path.join(img_path.split('.')[0] + '_log.jpg'),log)
    #return gam,log
def rotation(img_path, img_name, angle):
    img = Image.open(img_path)
    rotation_img = img.rotate(angle) #旋转角度
    rotation_img.save(os.path.join(FLAGS.img_save_path, img_name + '_rotation.jpg'))
    #rotation_img.save(os.path.join(img_path.split('.')[0] + '_rotation.jpg'))
    #return rotation_img

def randomGaussian(img_path, img_name, mean = 0, sigma = 1):  #高斯噪声
    image = Image.open(img_path)
    im = np.array(image)
    #设定高斯函数的偏移
    means = 0
    #设定高斯函数的标准差
    sigma = 25
    #r通道
    r = im[:,:,0].flatten()

    #g通道
    g = im[:,:,1].flatten()

    #b通道
    b = im[:,:,2].flatten()

    #计算新的像素值
    for i in range(im.shape[0]*im.shape[1]):

        pr = int(r[i]) + random.gauss(0,sigma)

        pg = int(g[i]) + random.gauss(0,sigma)

        pb = int(b[i]) + random.gauss(0,sigma)

        if(pr < 0):
            pr = 0
        if(pr > 255):
            pr = 255
        if(pg < 0):
            pg = 0
        if(pg > 255):
            pg = 255
        if(pb < 0):
            pb = 0
        if(pb > 255):
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb
    im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

    im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

    im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])
    gaussian_image = gaussian_image = Image.fromarray(np.uint8(im))
    gaussian_image.save(os.path.join(FLAGS.img_save_path, img_name + '_gaussian.jpg'))
    #gaussian_img.save(os.path.join(img_path.split('.')[0] + '_gaussian.jpg'))
    #return gaussian_image
def randomColor(img_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(img_path))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    dithering_img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    dithering_img.save(os.path.join(FLAGS.img_save_path, img_name + '_dithering.jpg'))
    #dithering_img.save(os.path.join(img_path.split('.')[0] + '_dithering.jpg'))
    #return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

def main(_):
    pp.pprint(FLAGS.__flags)

    print('You should specify the operation you want to process images, otherwise, nothing will be done.\n')
    print('You can use --translation=offset to translate images x axis translation(Translation fuction should not be use).\n \
               use --translation=offset twice can translate images both x and y axises. \n \
           --flip to filp images\n \
           --contrast to adjust the contrast of images\n \
           --rotation=angle to rotate images, default Rotate 90 degrees clockwise \n \
           --G_noise to add Gaussian noise into images\n \
           --Color_dithering to add color dithering into images\n')
    if not os.path.exists(FLAGS.images_dir):
        print('Please specify the direction for package, you can use --images_dir')
        return
    if not os.path.exists(FLAGS.img_save_path):
        os.makedirs(FLAGS.img_save_path)
    
 
    list = os.listdir(FLAGS.images_dir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
        image_path = os.path.join(FLAGS.images_dir,list[i])
        try:
            im = Image.open(image_path)
        except:
            print('Open %s error!\n'%(image_path))
            continue
        if FLAGS.translation:
            print('FLAG translation is specified, %s will be translated!->Store in %s\n'%(image_path, FLAGS.img_save_path))
            translation(image_path, list[i], FLAGS.translation)
        if FLAGS.flip:
            print('FLAG flip is specified, %s will be filpped!->Store in %s\n'%(image_path, FLAGS.img_save_path))
            flip(image_path, list[i])
        if FLAGS.contrast:
            print('FLAG contrast is specified, %s will be contrasted!->Store in %s\n'%(image_path, FLAGS.img_save_path))
            aj_contrast(image_path, list[i])
        if FLAGS.rotation:
            print('FLAG rotation is specified, %s will be rotated!->Store in %s\n'%(image_path, FLAGS.img_save_path))
            rotation(image_path, list[i], FLAGS.rotation)
        if FLAGS.G_noise:
            print('FLAG G_noise is specified, %s will be add Gaussian noise in!->Store in %s\n'%(image_path, FLAGS.img_save_path))
            randomGaussian(image_path, list[i])
        if FLAGS.Color_dithering:
            print('FLAG Color_dithering is specified, %s will be add Color dithering in!->Store in %s\n'%(image_path, FLAGS.img_save_path))
            randomColor(image_path, list[i])

if __name__ == '__main__':
    tf.app.run()
    main()
