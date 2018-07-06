#画直方图
import cv2 as cv
from matplotlib import pyplot as plt

def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])         #numpy的ravel函数功能是将多维数组降为一维数组
    plt.show()

def image_hist(image):     #画三通道图像的直方图
    color = ('b', 'g', 'r')   #这里画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i , color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])  #计算直方图
        plt.plot(hist, color)
        plt.xlim([0, 256])
    plt.show()

#图像直方图绘制
def Grayscale_histogram_1(img):
    img=plt.imread(img)

    plt.figure(img)
    plt.subplot(211)
    plt.imshow(img)
    arr=img.flatten()
    plt.subplot(212)
    n, bins, patches = plt.hist(arr, bins=256, normed=1)  
    plt.show()

def Grayscale_histogram_2(img, save_path):
    img=np.array(Image.open(img).convert('L'))

    plt.figure(img)
    arr=img.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)  
    cv.imwrite(save_path + ".png", n)
    plt.show()

def Color_histogram(img):
    src=Image.open(img)
    r,g,b=src.split()
    
    plt.figure()
    ar=np.array(r).flatten()
    plt.hist(ar, bins=256, normed=1,facecolor='r',edgecolor='r',hold=1)
    ag=np.array(g).flatten()
    plt.hist(ag, bins=256, normed=1, facecolor='g',edgecolor='g',hold=1)
    ab=np.array(b).flatten()
    plt.hist(ab, bins=256, normed=1, facecolor='b',edgecolor='b')
    plt.show()
'''
图像直方图均值化，如果一副图像的像素占有很多的灰度级而且分布均匀，那么这样的图像往往有高对比度和多变的灰度色调。直方图均衡化就是一种能仅靠输入图像直方图信息自动达到这种效果的变换函数。它的基本思想是对图像中像素个数多的灰度级进行展宽，而对图像中像素个数少的灰度进行压缩，从而扩展取值的动态范围，提高了对比度和灰度色调的变化，使图像更加清晰
'''
def exposure_equalize_hist(img):
    plt.figure("hist",figsize=(8,8))
    
    arr=img.flatten()
    plt.subplot(221)
    plt.imshow(img,plt.cm.gray)  #原始图像
    plt.subplot(222)
    plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red') #原始图像直方图
    
    img1=exposure.equalize_hist(img)
    arr1=img1.flatten()
    plt.subplot(223)
    plt.imshow(img1,plt.cm.gray)  #均衡化图像
    plt.subplot(224)
    plt.hist(arr1, bins=256, normed=1,edgecolor='None',facecolor='red') #均衡化直方图
    
    plt.show()

#type==0 处理图片;type==1 处理图片经过二值化的矩阵,
#type==1 画出来的是对黑色像素的统计
def horizontal_projection(img, imgisarray, type, save_dir=None):
    if imgisarray:  
        thresh1 = img
    else:
        img=cv.imread(img)  
        GrayImage=cv.cvtColor(img,cv.COLOR_BGR2GRAY)  
        ret,thresh1=cv.threshold(GrayImage,50,255,cv.THRESH_BINARY)  
    #print(thresh1[0,0])
    (x,y)=thresh1.shape  
    a = [0 for z in range(0, x)]  
    #print(a[0])
    #先将屏幕全部白(亮)化，再计算每行对应的所有黑像素点的总数.
    for i in range(0,x):  
        for j in range(0,y):  
            if  thresh1[i,j]==0:  
                a[i]=a[i]+1
                thresh1[i,j]=255    #to be white    
        #print(j)
    #将对应的像素涂黑
    for i  in range(0,x):  
        for j in range(0,a[i]):  
            thresh1[i,j]=0  

    #print(x,y)
     
    #cv.imshow("horizontal projection", thresh1)
    plt.imshow(thresh1,cmap=plt.gray())
    plt.title('horizontal projection')
    plt.show()
    #plt.imshow('horizontal projection',thresh1)

    if save_dir:
        cv.imwrite(save_dir, thresh1)

def vertical_projection(img, type, save_dir=None):
    if type == 0:  
        img=cv.imread(img)  
        GrayImage=cv.cvtColor(img,cv.COLOR_BGR2GRAY)  
        ret,thresh1=cv.threshold(GrayImage,50,255,cv.THRESH_BINARY)  
    elif type ==1:
        thresh1 = img

    # print(thresh1[0,0])#250  输出[0,0]这个点的像素值  返回值ret为阈值  
    # print(ret)#130  

    (h,w)=thresh1.shape #返回高和宽  
    # print(h,w)#s输出高和宽  
    a = [0 for z in range(0, w)]   
    #a = [0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数    
    #print(a)

    #记录每一列的波峰  
    for j in range(0,w): #遍历一列   
        for i in range(0,h):  #遍历一行  
            if  thresh1[i,j]==0:  #如果改点为黑点  
                a[j]+=1         #该列的计数器加一计数  
                thresh1[i,j]=255  #记录完后将其变为白色   
        # print (j)
      
    for j  in range(0,w):  #遍历每一列  
        for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑  
            thresh1[i,j]=0   #涂黑  

    #此时的thresh1便是一张图像向垂直方向上投影的直方图  
    #如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息  
      
    #cv.imshow("vertical projection", thresh1)
    plt.imshow(thresh1,cmap=plt.gray())  
    plt.title('vertical projection')
    plt.show()  
    if save_dir:
        cv.imwrite(save_dir, thresh1)

#图像腐蚀, 用于消除物体外的亮点
def erode(img, type, save_dir=None):
    if type == 0:  
        img=cv.imread(img)  
        GrayImage=cv.cvtColor(img,cv.COLOR_BGR2GRAY)  
        #ret,thresh1=cv.threshold(GrayImage,50,255,cv.THRESH_BINARY)  
        adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    elif type ==1:
         daptive_threshold = img

    element = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    adaptive_threshold_erode = cv.erode(adaptive_threshold, element, iterations = 1)

    plt.imshow(adaptive_threshold_erode, cmap=plt.gray())  
    plt.title('Image by erode')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "erode_img.png"), adaptive_threshold_erode)

#图像膨胀，用于消除物体内的亮点
def dilation(img, type, save_dir=None):
    if type == 0:  
        img=cv.imread(img)  
        GrayImage=cv.cvtColor(img,cv.COLOR_BGR2GRAY)  
        #ret,thresh1=cv.threshold(GrayImage,50,255,cv.THRESH_BINARY)  
        adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    elif type ==1:
        adaptive_threshold = img

    element = cv.getStructuringElement(cv.MORPH_RECT, (24, 6))
 
    adaptive_threshold_dilation = cv.dilate(adaptive_threshold, element, iterations = 1)
    
    plt.imshow(adaptive_threshold_dilation, cmap=plt.gray())  
    plt.title('Image by dilation')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "dilation_img.png"), adaptive_threshold_dilation)

#use canny edge detection
def edge_detect(img_path, save_dir=None):
    img = cv.imread(image_path)
    #高斯模糊,降低噪声
    blurred = cv.GaussianBlur(img,(3,3),0)

    cv.imshow("%s edge detect"%(fileName), blurred)

    #灰度图像
    gray=cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    #图像梯度
    xgrad=cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad=cv.Sobel(gray,cv.CV_16SC1,0,1)
    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_detect=cv.Canny(xgrad,ygrad,50,150)
    
    plt.imshow(edge_detect, cmap=plt.gray())  
    plt.title('Image edge detect')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "edge_detect.png"), edge_detect)

def text_edge_detect_1(img, save_path=None):

    img = cv.GaussianBlur(img,(3,3),0)
    canny = cv.Canny(img, 80, 170)
 
    plt.imshow(canny)  
    plt.title('Image edge detect')
    plt.show()  
    if save_path:
        cv.imwrite(os.path.join(save_dir, "edge_detect", ".png"), canny)

def print_pixels_value_PIL(img_path):
    img = Image.open(img_path)
    print (img.size)
    
    width = img.size[0]
    height = img.size[1]

    for x in range(0,width):
        for y in range(0,height):
            data = (img.getpixel((x,y)))

            #打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            print (type(data))
            print (data)

def print_pixels_value_CV(img_path):
    img = cv.imread(img_path)
    print (img.shape)
    
    width = img.shape[0]
    height = img.shape[1]

    for x in range(0,width):
        for y in range(0,height):
            #打印每个像素点的颜色RGBA的值(b,g,r)
            print ('(%d, %d): '%(x, y), (img[x, y]))

#color use rgb
def extract_pixels_specify_color(img_path, color, save_path=None):
    img = Image.open(img_path)
    print (img.size)
    print (img.getpixel((4,4)))
    
    width = img.size[0]
    height = img.size[1]

    for x in range(0,width):
        for y in range(0,height):
            data = (img.getpixel((x,y)))

            #打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            print (data)
            #将所有不与该颜色相同的像素点变成黑色
            if (data != color):
                img.putpixel((x,y),(0,0,0,255))
    #把图片强制转成RGB
    img = img.convert("RGB")
    #保存修改像素点后的图片
    img.save(save_path)


