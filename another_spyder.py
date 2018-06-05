#!/usr/bin/python  
# coding: utf-8
import urllib.parse
from urllib.request import urlopen
import json
import socket
import os  
import sys
import re
import argparse
import importlib
import string

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
# 设置 下载选项
add_arg('--keyword', '-kw', default='pretty girl', help='输入关键字')
add_arg('--download_page', '-dp', default=1, type=int, help='希望下载的页数(其中每页60张图)')
add_arg('--dir', default='./Images/', help='输入图片存放地址')
args = parser.parse_args()

valid_type = ['.png', '.jpg', '.PNG', '.JPG', '.gif', '.GIF', '.jpeg', '.JPEG']
download_page = args.download_page
socket.setdefaulttimeout(10)

# 设置 关键字 字符串
keyword = args.keyword

# 从百度图片爬取
tmpurl = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyword + '&pn='
# 从谷歌图片爬取(没成功，估计是开始是谷歌图片的网页设置和百度图片的网页设置有一些不同吧，比如，每页放的图片数？)
# tmpurl = 'https://www.google.com/search?safe=active&biw=1855&bih=964&tbm=isch&sa=1&ei=Mb8TWpTwJIGJ0wST-LWQDQ&q=aerial+photography&oq=' + '+'.join(list(keyword.split())) + '&gs_l=psy-ab.12..0j0i30k1l9.239767.241842.0.243293.4.4.0.0.0.0.1357.2234.2-1j5-1j0j1.3.0....0...1c.1j4.64.psy-ab..1.3.2234...0i7i30k1j0i13k1j0i13i30k1.0.I_kBiFEWvts'
# assert tmpurl == 'https://www.google.com/search?safe=active&biw=1855&bih=964&tbm=isch&sa=1&ei=Mb8TWpTwJIGJ0wST-LWQDQ&q=aerial+photography&oq=aerial+photography&gs_l=psy-ab.12..0j0i30k1l9.239767.241842.0.243293.4.4.0.0.0.0.1357.2234.2-1j5-1j0j1.3.0....0...1c.1j4.64.psy-ab..1.3.2234...0i7i30k1j0i13k1j0i13i30k1.0.I_kBiFEWvts'

s_dir = args.dir
try:
    if not os.path.isdir(s_dir):
        os.mkdir(s_dir)
except OSError:
    print('Can not make s_dir!')
    sys.exit()
i = 0

# 开始批量爬取图片
for dp in range(download_page):
    #urlopen解析中文出问题，需用以下quote解析中文
    url = urllib.parse.quote(tmpurl+str(dp*60), safe=string.printable)
    #url = tmpurl+str(dp*60)
    pattern = re.compile(r'setData\(\'imgData\', (\{[\s\S]*?\})\);')
    try:
        ipdata = urlopen(url).read().decode('utf-8')
    except IOError:
        print('can not open this url!')
        continue
        #sys.exit() 
    ipdata = pattern.search(ipdata)
    ipdata = ipdata.group(1)
    _regex = re.compile(r'\\(?![/u"])')  
    ipdata = _regex.sub(r"\\\\", ipdata) 
    imgData = json.loads(ipdata, strict=False)

    if imgData['data']:
        for obj in imgData['data']:
            if obj and obj['objURL']:
                try:
                    data_img = urlopen(obj['objURL']).read()
                except IOError:
                    print('--- Meet damaged image.')
                else:
                    fPostfix = os.path.splitext(obj['objURL'])[1]
                    if fPostfix in valid_type:
                        filename = os.path.join(s_dir, os.path.basename(obj['objURL']))
                    else:
                        filename = os.path.join(s_dir, os.path.basename(obj['objURL'])) + '.jpg'
                    try:
                        file_obj = open(filename, 'wb')
                        file_obj.write(data_img)
                        file_obj.close()
                    except socket.timeout:
                        print('socket time out!')
                    else:
                        i += 1
                        print('+++ Img '+ str(i) + ' is downloaded .')
                    finally:
                        pass

print('All images have been downloaded!')
