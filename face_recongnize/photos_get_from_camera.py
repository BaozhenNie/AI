#-*- coding: utf-8 -*-

import cv2 as cv
import sys
from PIL import Image

def CatchPICFromVideo(window_name, camera_addr, catch_pic_num, pic_save_path):
    cv.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv.VideoCapture(camera_addr)        
      
    #告诉OpenCV使用人脸识别分类器
    classfier = cv.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
           
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
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
                #将当前帧保存为图片
                img_name = '%s/%d.jpg'%(pic_save_path, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                    break
                
                #画出矩形框
                cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                 
                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        #超过指定最大保存数量结束程序
        if num > (catch_pic_num): break                
                        
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
    if len(sys.argv) != 4:
        print("Usage:%s camera_addr face_num_max pic_save_path\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("截取人脸", sys.argv[1], int(sys.argv[2]), sys.argv[3])
