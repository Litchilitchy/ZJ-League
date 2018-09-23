# -*- coding: cp936 -*-
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from queue import PriorityQueue as PQueue
import cv2


# ����ͼ��
def regularizeImage(img, size = (8, 8)):
    return img.resize(size).convert('L')


# ����hashֵ
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


# �Ƚ�hashֵ
def compHashCode(hc1, hc2):
    cnt = 0
    for i, j in zip(hc1, hc2):
        if i == j:
            cnt += 1
    return cnt


# ����ƽ����ϣ�㷨���ƶ�
def calaHashSimilarity(img1, img2):
    img1 = regularizeImage(img1)
    img2 = regularizeImage(img2)
    hc1 = getHashCode(img1)
    hc2 = getHashCode(img2)
    return compHashCode(hc1, hc2)


#ѡ���ؼ�֡
'''
def getKeyFrame(name):
  keyFrame = PQueue()
  threshold = -1
  files = os.listdir(imgPath)
  img1 = None
  #������н�ͼ��ѡ��3��
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
        
  #��3�Źؼ�֡���Ƶ�output������֮ǰ�����н�ͼɾ��      
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
#ÿ8֡��ȡͼƬ���������������3����outputPath��������ʽ����Ƶ����_1.jpg
#Ϊ���������룬�˴���ƵĬ��û����1000֡��
  videos = os.listdir(dataPath)

  for v in videos:
    name = v.split('.')[0]
    print(v)
    #ÿ8֡��ȡͼƬ
    vc = cv2.VideoCapture(dataPath + '/' + v) #������Ƶ�ļ�
    rval = False
    frame_num = 0
    if vc.isOpened(): #�ж��Ƿ�������
        rval, frame = vc.read()
        frame_num = vc.get(7)
        print(frame_num)
    else:
        rval = False
    
    isVid = rval
    count = 0
    num = frame_num /  6
    num = int(num) + 1
    #��ȡ5�Źؼ�֡

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
