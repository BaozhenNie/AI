# coding:utf-8
import sys
import os
import cv2 as cv
import pprint

#import this to use flags
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("images_dir", "./images", "Directory name saving the origin images need to preprocessing.")
flags.DEFINE_string("img_save_path", "./images_by_processing", "Directory name to save the processing images.")
flags.DEFINE_integer("crop_save_pixels", 25, "pixels beside the faces you want to reserve[x,y,w,h]")
flags.DEFINE_string("opencv_ins_dir", r"/home/niebaozhen/anaconda3/pkgs/opencv3-3.1.0-py36_0/", "Directory that OpenCV installed[directory should be given entirely to opencvx-x.x.x*].")
FLAGS = flags.FLAGS

image_num = 0

pp = pprint.PrettyPrinter()
#main argument '_' could not be omitted
def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.images_dir):
        print('Please specify the direction for package, you can use --images_dir')
        return
    if not os.path.exists(FLAGS.img_save_path):
        os.makedirs(FLAGS.img_save_path)
    
    # 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
    classifier_dir = os.path.join(FLAGS.opencv_ins_dir, 'share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier(classifier_dir)
    
    list = os.listdir(FLAGS.images_dir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           image_path = os.path.join(FLAGS.images_dir,list[i])
           if os.path.isfile(image_path):
                # 读取图片
                image = cv.imread(image_path)
                if image is None:
                    continue
                gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
                
                # 探测图片中的人脸
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor = 1.15,
                    minNeighbors = 5,
                    minSize = (5,5),
                    flags = cv.CASCADE_SCALE_IMAGE
                )

                if len(faces) > 0:
                    print('第{0}张图片发现{1}个人脸!'.format(i, len(faces)))
    
                    for (x,y,w,h) in faces:
                        global image_num
                        p = FLAGS.crop_save_pixels
                        img_name = '%s/%d.jpg'%(FLAGS.img_save_path, image_num)
                        image = image[y - p : y + h + p, x - p : x + w + p]
                        cv.imwrite(img_name, image)

                        image_num += 1
                        print('Save image, name = %s, image_num = %d'%(img_name, image_num))

if __name__ == '__main__':
    tf.app.run()
    main()
