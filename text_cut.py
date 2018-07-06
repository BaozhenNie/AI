# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import pprint
#import this to use flags
import tensorflow as tf

import img_utils as utl 

flags = tf.app.flags
flags.DEFINE_string("base_dir", "./origin/", "Directory saving the origin images need to preprocessing.")
flags.DEFINE_string("dst_dir", "./result/", "Directory saving the result images.")
flags.DEFINE_integer("min_val", 10, "Use to cutting text, if horizontal_sum > min_val, we take this line as this start.")
flags.DEFINE_integer("min_range", 30, "Use to cutting text, means the length of text column must > min_range pixels")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

base_dir = FLAGS.base_dir
dst_dir = FLAGS.dst_dir
min_val = FLAGS.min_val
min_range = FLAGS.min_range
count = 0

def extract_peek(array_vals, minimun_val, minimun_range):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            if i - start_i >= minimun_range:
                end_i = i
                print(end_i - start_i)
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def cutImage(img, peek_ranges, vertical_peek_ranges2d):
    global count
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            count += 1
            img1 = img[y:peek_range[1], x:vertical_range[1]]
            new_shape = (150, 150)
            img1 = cv2.resize(img1, new_shape)
            cv2.imwrite(dst_dir + str(count) + ".png", img1)
            # cv2.rectangle(img, pt1, pt2, color)

def main(_):
    pp.pprint(FLAGS.__flags)
    for fileName in os.listdir(base_dir):
        fileName_process = fileName.split('.')[0]
        #fileName_process_dir used to save the images that be processed
        fileName_process_dir = os.path.join(dst_dir, fileName_process)
        if not os.path.exists(fileName_process_dir):
            os.makedirs(fileName_process_dir)
        filepath = os.path.join(base_dir, fileName)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        #cv2.namedWindow("The image after adaptive threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("Adaptive threshold img", adaptive_threshold)

        #Write an image to save the image after adaptive threshold
        cv2.imwrite(os.path.join(fileName_process_dir, "Adaptive_threshold.png"), adaptive_threshold)

        #image_horizontal_projection used to save the image that be horizontal_projected
        image_horizontal_projection = np.copy(adaptive_threshold)
        utl.horizontal_projection(image_horizontal_projection, 1, os.path.join(fileName_process_dir, "horizontal_projection", ".png"))

        #Compute the image histogram on the rows by sum all values of each column 
        horizontal_sum = np.sum(adaptive_threshold, axis=1)
        #Compute the location that might be the start and end of texts
        peek_ranges = extract_peek(horizontal_sum, min_val, min_range)
        print('Find %d rows for text\n'%(len(peek_ranges)))
        #line_seg_adaptive_threshold used to save the image that be rectangled
        line_seg_adaptive_threshold = np.copy(adaptive_threshold)
        for i, peek_range in enumerate(peek_ranges):
            x = 0
            y = peek_range[0]
            w = line_seg_adaptive_threshold.shape[1]
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)

        cv2.imshow("%s be rectangled"%(fileName), line_seg_adaptive_threshold)
        cv2.imwrite(os.path.join(fileName_process_dir, "rectangle.png"), line_seg_adaptive_threshold)

        vertical_peek_ranges2d = []

        for i, peek_range in enumerate(peek_ranges):
            start_y = peek_range[0]
            end_y = peek_range[1]
            line_img = adaptive_threshold[start_y:end_y, :]

            cv2.imshow("line image", line_img)
            
            utl.vertical_projection(line_img, 1, os.path.join(fileName_process_dir, "vertical_projection_%d"%i, ".png"))

            vertical_sum = np.sum(line_img, axis=0)
            vertical_peek_ranges = extract_peek(
                vertical_sum, min_val, min_range)
            vertical_peek_ranges2d.append(vertical_peek_ranges)
        cutImage(img, peek_ranges, vertical_peek_ranges2d)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    tf.app.run()
    main()
