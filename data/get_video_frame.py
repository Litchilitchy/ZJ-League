# -*- coding: UTF-8 -*-


# 正则化图像
def regularizeImage(img, size = (8, 8)):
    return img.resize(size).convert('L')


# 计算hash值
def getHashCode(img, size = (8, 8)):
 
    pixel = []
    for i in range(size[0]):
        for j in range(size[1]):
            pixel.append(img.getpixel((i, j)))
 
    mean = sum(pixel) / len(pixel)
 
    result = []
    for i in pixel:
        if i > mean:
            result.append(1)
        else:
            result.append(0)
    
    return result


# 比较hash值
def compHashCode(hc1, hc2):
    cnt = 0
    for i, j in zip(hc1, hc2):
        if i == j:
            cnt += 1
    return cnt


# 计算平均哈希算法相似度
def calaHashSimilarity(img1, img2):
    img1 = regularizeImage(img1)
    img2 = regularizeImage(img2)
    hc1 = getHashCode(img1)
    hc2 = getHashCode(img2)
    return compHashCode(hc1, hc2)


#选出关键帧
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from queue import PriorityQueue as PQueue

#-----------------------------------------------路径修改一下
#imgPath = "datalab/2193/data/tempImg/"
outputPath = "datalab/2193/data/output/"

def getKeyFrame(name):
  keyFrame = PQueue()
  threshold = -1
  files = os.listdir(imgPath)
  img1 = None
  #检测所有截图，选出3张
  for file in files:
    if (keyFrame.empty()):
      keyFrame.put((threshold, file))
      img1 = Image.open(imgPath + file)
    else:
      img2 = Image.open(imgPath + file)
      sim = 0 - calaHashSimilarity(img1, img2)/64
      if (sim > threshold):
        if (keyFrame.qsize() == 3):
          keyFrame.get()
          threshold = sim
        keyFrame.put((sim, file))
        img1 = img2
        
  #把3张关键帧复制到output，并把之前的所有截图删除      
  count = 1
  while not keyFrame.empty():
    tmp = keyFrame.get()[1]
    print(tmp)
    shutil.copy(imgPath + tmp, outputPath + name + '_' + str(count) + '.jpg')
    count = count + 1
    
  for file in files:
    os.remove(imgPath + file)


#每8帧提取图片，保存符合条件的3张在outputPath，命名格式：视频名称_1.jpg
#为了命名整齐，此处视频默认没超过1000帧的
import cv2

dataPath = 'train'
videos = os.listdir(dataPath)

if not os.path.exists(dataPath + '/tempImg'):
  os.mkdir(dataPath + '/tempImg')
if not os.path.exists(dataPath + '/output'):
  os.mkdir(dataPath + '/output')

#imgPath = 'datalab/2193/data/tempImg/'
outputPath = 'datalab/2193/data/output/'

for v in videos:
  name = v.split('.')[0]
  print(v)
  #每8帧提取图片
  vc = cv2.VideoCapture(dataPath + '/' + v) #读入视频文件
  c=1
  rval = False
  if vc.isOpened(): #判断是否正常打开
      rval, frame = vc.read()
  else:
      rval = False
  isVid = rval
  timeF = 8  #视频帧计数间隔频率
  while rval:   #循环读取视频帧
      #print("-----"+str(c))
      rows, cols, channel = frame.shape
      frame=cv2.resize(frame,(224, 224),fx=0,fy=0,interpolation=cv2.INTER_AREA)
      if(c%timeF == 0): #每隔timeF帧进行存储操作
          print("--write-to-file---: " + str(c))
          s = ""
          if (c < 100):
              s = s + "0"
          if (c < 10):
              s = s + "0"
          cv2.imwrite(imgPath + name + '_' + s + str(c) + '.jpg',frame) #存储为图像在imgPath,视频名称_000.jpg
      c = c + 1
      rval, frame = vc.read()
  vc.release()
  
  #提取3张关键帧
  if isVid:
    getKeyFrame(name)
