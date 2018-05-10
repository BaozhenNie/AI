#coding=utf-8  
import cv2 as cv  
  
import time  
  
if __name__ == '__main__':  
  
    cv.namedWindow("camera",1)  
    #开启ip摄像头  
    video="http://admin:admin@172.16.200.214:8081/"  
    capture =cv.VideoCapture(video)  
    
    num = 0;  
    while True:  
        img = cv.QueryFrame(capture)  
        cv.ShowImage("camera",img)  
  
        #按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口  
        key = cv.WaitKey(10)   
  
        if key == 27:  
            #esc键退出  
            print('esc break...')
            break
        if key == ord(' '):  
            #保存一张图像  
            num = num+1  
            filename = "frames_%s.jpg" % num  
            cv.SaveImage(filename,img)  
  
    del(capture)  
    cv.DestroyWindow("camera")
