#-*- coding: utf-8 -*-

import cv2 as cv
import sys
from PIL import Image

def CatchVideo(window_name, camera_addr):
    cv.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv.VideoCapture(camera_addr)        
      
    #告诉OpenCV使用人脸识别分类器
    classfier = cv.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
           
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
 
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                    
        
	#将当前帧转换成灰度图像
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                 
 
        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:            #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect        
                cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                        
        #显示图像
                        
        #显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv.imshow(window_name, frame)
        c = cv.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_addr\r\n" % (sys.argv[0]))
    else:
        CatchVideo("截取视频流", str(sys.argv[1]))
