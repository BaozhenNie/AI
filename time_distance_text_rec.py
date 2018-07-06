#coding=utf-8  
from PIL import Image
import pytesseract
import cv2 as cv
import sys

def time_dist_text_rec(image_path):

    img = Image.open(image_path)
    img_size = img.size
    time_region = img.crop((35,1025,430,1065))
    dist_region = img.crop((920,1025,1030,1065))
    time_region.save('./time_region.jpeg')
    dist_region.save('./dist_region.jpeg')
   
    time =  pytesseract.image_to_string(time_region)
    dist =  pytesseract.image_to_string(dist_region)
    return time, dist

def main(argv):
   if len(argv) == 1:
      print('You should specify the path of the image that you want to recongnize.\n \
      Usage: text_recongnize.py xxx/xx.img.')
   print(time_dist_text_rec(argv[1]))

if __name__ == '__main__':
    main(sys.argv)
