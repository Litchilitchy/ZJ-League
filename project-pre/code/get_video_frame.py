# -*- coding: cp936 -*-
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from queue import PriorityQueue as PQueue
import cv2


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
'''
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
      print(sim)
      if (sim >= threshold):
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
'''
def getFrame(dataPath, outputPath):
#每8帧提取图片，保存符合条件的3张在outputPath，命名格式：视频名称_1.jpg
#为了命名整齐，此处视频默认没超过1000帧的
  videos = os.listdir(dataPath)

  for v in videos:
    name = v.split('.')[0]
    print(v)
    #每8帧提取图片
    vc = cv2.VideoCapture(dataPath + '/' + v) #读入视频文件
    rval = False
    frame_num = 0
    if vc.isOpened(): #判断是否正常打开
        rval, frame = vc.read()
        frame_num = vc.get(7)
        print(frame_num)
    else:
        rval = False
    
    isVid = rval
    count = 0
    num = frame_num /  6
    num = int(num) + 1
    #提取5张关键帧

    if isVid:
      s = 0
      while rval:
        if rval and count % num == 0 and count != 0 and count != frame_num:
          cv2.imwrite(outputPath + name + '_' + str(s) + '.jpg',frame)
          s = s + 1
          print("save " + name + "_" + str(count / num))
        rval, frame = vc.read()
        count = count + 1
      vc.release()
      #getKeyFrame(name)
    

if __name__ == "__main__":
  testPath = './../data/test'
  trainPath ='./../data/train'

  if not os.path.exists('./../data/test_img'):
    os.mkdir('./../data/test_img')
  if not os.path.exists('./../data/train_img'):
    os.mkdir('./../data/train_img')

  testImgPath = './../data/test_img'
  trainImgPath = './../data/train_img'

  getFrame(testPath, testImgPath)
  getFrame(trainPath, trainImgPath)
