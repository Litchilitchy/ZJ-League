# ZJ-League
ZJ杯 视频识别

#### 目录结构
    ---
      _instruction   # 用来放说明文件
      _test_function   # 用来测试单个模块 
      feature
        ---extract_image.py 
        ---extract_question.py  
      model.py  
      train.py   
      data_iter.py
      
      _v1
      _v2
      ...
      
以上为github中目录，使用时需先copy以上目录到项目路径，另外，实际使用目录变更如下：

需要新增一个glove路径用于存放词向量模型，具体方法为，在`feature`文件夹下，放入`glove.6B.zip`（建议在 http://nlp.stanford.edu/data 中下载并复制，使用python中的下载函数容易出错，并且不支持断点）

需要用到视频关键帧截取`get_video_frame.py`（此模块在`data`文件夹下），输出图片（每个视频三张）作为训练数据，和原训练文本数据一起放在根目录下的文件夹中

最初的模型从 v1 -> vn 版本存放在 \_v1 -> \_vn 文件夹中，根目录下为最终模型，v1 -> vn 每个文件夹下都包含所有该模型代码（`feature`中的两个`.py`和根目录下的三个`.py`）

实际使用目录如下，github上的data目录中包含了截取的非常少量的数据，仅用于debug跑通程序
    
    ---   
    data        
      train_img
        ---vid1_1.jpg
        ---vid1_2.jpg
        ...
        ---vidn_3.jpg
      test_img
        ---vid1_1.jpg
        ---vid1_2.jpg
        ...
        ---vidn_3.jpg    
      ---train.txt
      ---test.txt
      ---submit.txt
      get_video_frame.py    # this .py is not necessary for program running
    feature
      glove_model
        ---glove.6B.zip
      ---extract_image.py 
      ---extract_question.py  
    model.py
    train.py
    data_iter.py
      

#### 训练方法
`feature`文件夹中的`extract_image`和`extract_question`可以单独运行，先分别运行一次，生成`train.py`中需要的`.npy`文件，作为提取特征的预训练

`model`为模型，`data_iter`中为读取预训练生成的`.npy`数据的方法

运行结束`feature`文件夹中的代码后，运行`train.py`生成answer数据输出
      
### 方法
      
#### 视频关键帧提取      
   
#### image-question模型

![image](https://github.com/SummerLitchy/Zcup/blob/master/_instruction/VQA-attention.png)

先直接搬来一个模型 https://github.com/shiyangdaisy23/vqa-mxnet-gluon/blob/master/Attention-VQA-gluon.ipynb

以上这个模型作为最终参考版本，由于这个模型有一定的复杂度，我们不直接上它，下面的模型从比较简单的开始实验，并做记录

##### v1 模型
采用`shape=(2048,)`的一维图片向量，三张图片通过拼接得到`shape=(6144,)`和`shape=(100,)`一维词向量

**Time:** 

extract image feature (`extract_image.py`): 13 min per 500 images

extract question feature (`extract_question.py`): 比图片提取快很多

train (`train.py`): 主要消耗为训练时间，相比之下，load data 和 predict 的时间可以忽略不计

时间优化：1.并行处理图片特征

**Accuracy:**

数据缺失时容易出BUG，顺序读取数据（no index）的方式导致单个数据missing时所有数据乱序，难以debug，直接进入v2模型

##### v2 模型
采用`shape=(2048,)`的一维图片向量和`shape=(100,)`一维词向量，暂时没有对图片提取进行效率优化

**Accuracy:**

mini data test accuracy 22% after 10 epochs, training data accuracy 16% after 10 epochs


