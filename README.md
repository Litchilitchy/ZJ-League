# ZJ-League
Alibaba ZJ League Video Question Answering
 
Competition Link: https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.1122325cOBX6Rr&raceId=231676&_lang=en_US

初赛代码已按提交要求整理在 `project-pre` 中

Code of pre-competition is packaged in `project-pre` dir as the requirements of the competition

#### 目录结构 File Contents
    ---
      _instruction   # 用来放说明文件 
      data
        train
        test
        --train.txt
        --test.txt        
      code
        feature
          glove_model
          --extract_image.py 
          --extract_question.py 
        get_video_frame.py
        evaluate.py
        model.py  
        train.py   
        data_iter.py
        main.py  
      submit

#### 使用方法和说明 Run Instructions

所有代码在`code`文件夹中 All the code are in `code` directory

一步运行：将训练用的视频文件夹`train,test`和文本`train.txt,test.txt`放在`data`文件夹下，运行`code/main.py`即可得到结果在`submit`文件夹中

Run in one line code: put the training video dir `train, test` and training text file `train.txt,test.txt` in `data` dir, and run `code/main.py`, then the result will be generated in `submit` dir

另外，实际使用时需要加入以下操作：

Additionally, there are several more operations during official run

需要新增一个glove路径用于存放词向量模型，具体方法为，在`feature`文件夹下，创建文件夹`glove_model`，放入`glove.6B.zip`（建议在 http://nlp.stanford.edu/data 中下载并复制，使用python中的下载函数容易出错，并且不支持断点）

Make a new directory to store the glove model, in dir `feature/glove_model`, and copy the file `glove.6B.zip` into it. It is suggested to directly download it from website http://nlp.stanford.edu/data, since the download in python is easy to get interrupted and the breakpoint resume is not supported
    

#### 分步训练方法 Separated Training Steps
`feature`文件夹中的`extract_image`和`extract_question`可以单独运行，先分别运行一次，生成`train.py`中需要的`.npy`文件，作为提取特征的预训练

The program `extract_image.py` and `extract_question.py` can run seperately, first run each of them to get the feature `.npy` file, which will be needed in training steps

`model`为模型，`data_iter`中为读取预训练生成的`.npy`数据的方法

`model.py` is the model file, and `data_iter.py` contains the iterator to generated the data batch from the feature file `.npy`

运行结束`feature`文件夹中的代码后，运行`train.py`生成answer数据输出

Once get the `.npy` after running the `.py` program in `feature` dir, run `python train.py` to get the answer

### Flow-diagram

![img wrong](https://github.com/SummerLitchy/ZJ-League/raw/master/_instruction/ZJL_flow_diagram.png)

