#coding=utf-8  
from PIL import Image
import pytesseract
import cv2 as cv
import sys
import os
from matplotlib import pyplot as plt  

#图像膨胀，用于消除物体内的亮点
def dilation(img, save_dir=None):
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
 
    img_dilation = cv.dilate(img, element, iterations = 1)
    
    plt.imshow(img_dilation, cmap=plt.gray())  
    plt.title('Image by dilation')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "dilation_img.png"), img_dilation)

#search pixels that aroud specific location pixel is there white or black pixel
#loc: pixel location spacificed for compare
#img: 
#k: kernel size
is_white = lambda pixel: pixel[0]>220 and pixel[1]>220 and pixel[2]>220 
is_black = lambda pixel: pixel[0]<=20 and pixel[1]<=20 and pixel[2]<=20
def pixel_nearby_traverse_CV(img, loc, k=3):
    kx_s = loc[0] - k//2 if (loc[0] - k//2) > 0 else 0
    kx_e = loc[0] + k//2 if (loc[0] + k//2) < img.shape[0] else img.shape[0]
    ky_s = loc[1] - k//2 if (loc[1] - k//2) > 0 else 0
    ky_e = loc[1] + k//2 if (loc[1] + k//2) < img.shape[1] else img.shape[1]
    for k_x in range(kx_s, kx_e):
        for k_y in range(ky_s, ky_e):
            pixel = img[k_x, k_y]
            if(is_white(pixel) or is_black(pixel)):
                return True
    return False

def text_edge_extract_opt_CV(img_gray, img_orig, k=3, save_dir=None):
    print ('Image gray shape:', img_gray.shape)
    print ('Image origin shape:', img_orig.shape)

    width = img_gray.shape[0]
    height = img_gray.shape[1]

    for x in range(0,width):
        for y in range(0,height):

            if (img_gray[x,y] == 255):
                if not pixel_nearby_traverse_CV(img_orig, (x,y), k):
                    img_gray[x,y] = 0

    plt.imshow(img_gray, cmap=plt.gray())  
    plt.title('Image text edge extract optimize')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "text_edge_extract_optimize.png"), img_gray)

    return img_gray
#use canny edge detection
def text_edge_detect(img, save_dir=None):
    #高斯模糊,降低噪声
    blurred = cv.GaussianBlur(img,(3,3),0)

    cv.imshow("Image after GaussianBlurred", blurred)

    #灰度图像
    gray=cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    #图像梯度
    xgrad=cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad=cv.Sobel(gray,cv.CV_16SC1,0,1)
    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_detect=cv.Canny(xgrad,ygrad,70,160)

    plt.imshow(edge_detect, cmap=plt.gray())  
    plt.title('Image edge detect')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir,"edge_detect.png"), edge_detect)

    return edge_detect

def text_edge_detect_1(img, save_dir=None):

    img = cv.GaussianBlur(img,(3,3),0)
    canny = cv.Canny(img, 80, 170)
 
    plt.imshow(canny)  
    plt.title('Image edge detect')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "edge_detect", ".png"), canny)

def text_edge_detect_2(img, save_dir=None):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    plt.imshow(adaptive_threshold, cmap=plt.gray())  
    plt.title('Image noise filter')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "moise_filter", ".png"), adaptive_threshold)
    
    img_Blur= cv.GaussianBlur(adaptive_threshold,(3,3),0)
    canny = cv.Canny(img_Blur, 700, 1000)
 
    plt.imshow(canny)  
    plt.title('Image edge detect')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "edge_detect", ".png"), canny)


def noise_filter(img, save_dir):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    plt.imshow(adaptive_threshold, cmap=plt.gray())  
    plt.title('Image noise filter')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "moise_filter", ".png"), adaptive_threshold)

def text_recongnize(image_path, need_prepro=False):
    img = Image.open(image_path)
    print(type(img))
    if need_prepro:
        text_edge_detect(img, './')
        #text_edge_detect_1(img, './')
        #noise_filter(img, './')
        #text_edge_detect_2(img, './')
    return pytesseract.image_to_string(img)

def print_pixels_value(img_path):
    img = Image.open(img_path)
    print (img.size)
    print (img.getpixel((4,4)))

    width = img.size[0]
    height = img.size[1]

    for i in range(0,width):
        for j in range(0,height):
            data = (img.getpixel((i,j)))

            #print (type(data))
            print ('(%d, %d): '%(i, j), (data))

def main(argv):
   if len(argv) == 1:
      print('You should specify the path of image that you want to recongnize.\n \
      Usage: text_recongnize.py xxx/xx.img.')
   print(text_recongnize(argv[1], False))

   #img = cv.imread(argv[1])
   #edge_detect = text_edge_detect(img, './')
   #img_gray_opt = text_edge_extract_opt_CV(edge_detect, img, 5, './')
   #dilation(img_gray_opt, './')

   #gray2rgb = cv.cvtColor(img_gray_opt, cv.COLOR_GRAY2RGB)
   #cv.imwrite(os.path.join("gray_to_rgb.png"), gray2rgb)
   #print_pixels_value(argv[1])

if __name__ == '__main__':
    main(sys.argv)
