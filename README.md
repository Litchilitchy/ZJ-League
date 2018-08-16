# Zcup
ZJ杯视频识别

#### 目录结构
    ---
      _instruction   # 用来放说明文件
      feature
        ---extract_image.py 
        ---extract_question.py  
      model.py
      train.py

#### 训练方法
`feature`文件夹中的`extract_image`和`extract_question`可以单独运行，作为提取特征的预训练

运行结束`feature`文件夹中的代码后，运行`train.py`生成answer数据输出
      
#### 模型

##### 视频关键帧提取

##### image-question模型

![image](https://github.com/SummerLitchy/Zcup/blob/master/_instruction/VQA-attention.png)

先直接搬来这个模型 https://github.com/shiyangdaisy23/vqa-mxnet-gluon/blob/master/Attention-VQA-gluon.ipynb
