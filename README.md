# Zcup
ZJ杯视频识别

#### 目录结构
    ---
      _instruction   # 用来放说明文件
      _test_function   # 用来测试单个模块      
      feature
        ---extract_image.py 
        ---extract_question.py  
      model.py
      train.py
      
以上为github中目录，使用时需先copy以上目录到项目路径，另外，实际使用目录变更如下：

需要新增一个glove路径用于存放词向量模型，具体方法为，并且新建一个文件夹，放入`glove.6B.zip`（建议在 http://nlp.stanford.edu/data 中下载并复制，使用python中的下载函数容易出错，并且不支持断点）

需要用到视频关键帧截取（此模块不包括在主要代码中），输出图片（每个视频三张）作为训练数据，和原训练文本数据一起放在根目录下的文件夹中

最初的模型从 v1 -> vn 版本存放在 v1 -> vn 文件夹中，根目录下为最终模型

实际目录除上述外，新增如下
    
    ---
      data
        img
          ---vid1_1.jpg
          ---vid1_2.jpg
          ...
          ---vidn_3.jpg
        ---train.txt
        ---test.txt
        ---submit.txt

#### 训练方法
`feature`文件夹中的`extract_image`和`extract_question`可以单独运行，作为提取特征的预训练

运行结束`feature`文件夹中的代码后，运行`train.py`生成answer数据输出
      
### 方法
      
#### 视频关键帧提取      
   
#### image-question模型

![image](https://github.com/SummerLitchy/Zcup/blob/master/_instruction/VQA-attention.png)

先直接搬来一个模型 https://github.com/shiyangdaisy23/vqa-mxnet-gluon/blob/master/Attention-VQA-gluon.ipynb

以上这个模型作为最终参考版本，由于这个模型有一定的复杂度，我们不直接上它，下面的模型从比较简单的开始实验，并做记录

##### v1 模型
采用2048的一维图片向量和一维词向量
