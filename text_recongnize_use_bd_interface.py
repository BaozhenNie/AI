#coding=UTF-8  
  
from aip import AipOcr  
import json  
import sys
 
# 定义常量  
APP_ID = '11388905'  
API_KEY = 'lOZ73Aa8EZQI6oEi4dpcwVUO'  
SECRET_KEY = 'afCu4fUz023H40ORSxVKg4bqdk9b8Iyk'  

# 初始化AipFace对象
aipOcr = AipOcr(APP_ID, API_KEY, SECRET_KEY)  

# 读取图片
filePath = sys.argv[1]  
def get_file_content(filePath):  
    with open(filePath, 'rb') as fp:  
        return fp.read()  
  
# 定义参数变量  
options = {  
  'detect_direction': 'true',  
  'language_type': 'CHN_ENG',  
}  
  
# 调用通用文字识别接口  
result = aipOcr.basicGeneral(get_file_content(filePath), options)  
print(result)


