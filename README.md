# ZJ-League
Alibaba ZJ League Video Question Answering
 
Competition Link: https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.1122325cOBX6Rr&raceId=231676&_lang=en_US

初赛代码已按提交要求整理在 `project-pre` 中，主目录整理好后会删除旧版本备份

Code of pre-competition is packaged in `project-pre` dir as the requirements of the competition

#### 目录结构 File Contents
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

Above is the contents on github, before official run, first copy the contents above to your project dir. Additionally, there are several more operations during official run

需要用到视频关键帧截取`get_video_frame.py`（此模块在`data`文件夹下），输出图片（每个视频自定义张数）作为训练数据，和原训练文本数据一起放在根目录下的文件夹中，并命名为`train_img, test_img`

First of all, use `get_video_frame.py` (locate in `data` dir) to extract some images from the source training video, and output the images to `train_img, test_img` at the project dir

需要新增一个glove路径用于存放词向量模型，具体方法为，在`feature`文件夹下，创建文件夹`glove_model`，放入`glove.6B.zip`（建议在 http://nlp.stanford.edu/data 中下载并复制，使用python中的下载函数容易出错，并且不支持断点）

Make a new directory to store the glove model, in dir `feature/glove_model`, and copy the file `glove.6B.zip` into it. It is suggested to directly download it from website http://nlp.stanford.edu/data, since the download in python is easy to get interrupted and the breakpoint resume is not supported

最初的模型从 v1 -> vn 版本存放在 \_v1 -> \_vn 文件夹中，根目录下为最终模型，v1 -> vn 每个文件夹下都包含所有该模型代码（`feature`中的两个`.py`和根目录下的三个`.py`）

实际使用目录如下，github上的data目录中包含了截取的非常少量的数据，仅用于debug跑通程序

The overall contents is as following, and `data` dir on github only contains very few training data, only used to make a run for your program
    
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
      

#### 训练方法 Training Steps
`feature`文件夹中的`extract_image`和`extract_question`可以单独运行，先分别运行一次，生成`train.py`中需要的`.npy`文件，作为提取特征的预训练

The program `extract_image.py` and `extract_question.py` can run seperately, first run each of them to get the feature `.npy` file, which will be needed in training steps

`model`为模型，`data_iter`中为读取预训练生成的`.npy`数据的方法

`model.py` is the model file, and `data_iter.py` contains the iterator to generated the data batch from the feature file `.npy`

运行结束`feature`文件夹中的代码后，运行`train.py`生成answer数据输出

Once get the `.npy` after running the `.py` program in `feature` dir, run `python train.py` to get the answer
      
### 方法    
   
#### image-question模型

![image](https://github.com/SummerLitchy/Zcup/blob/master/_instruction/VQA-attention.png)

来源于 https://github.com/shiyangdaisy23/vqa-mxnet-gluon/blob/master/Attention-VQA-gluon.ipynb

### log
v1: can not be used

v2: word emd only, 1 image per video, concat

v3: to be added

v4: lstm added

v5: evaluate added

v6: change concat to element_mul, cross validation added

v7: 5 image per video, lstm


