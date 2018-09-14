##### v1 模型 (abandoned)
采用`shape=(2048,)`的一维图片向量，三张图片通过拼接得到`shape=(6144,)`和`shape=(100,)`一维词向量

**Time:** 

extract image feature (`extract_image.py`): 13 min per 500 images

extract question feature (`extract_question.py`): 比图片提取快很多

train (`train.py`): 主要消耗为训练时间，相比之下，load data 和 predict 的时间可以忽略不计

时间优化：1.并行处理图片特征

**Accuracy:**

数据缺失时容易出BUG，顺序读取数据（no index）的方式导致单个数据missing时所有数据乱序，难以debug，直接进入v2模型
