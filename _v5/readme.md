##### v5 模型

input feature: image feature (bs, 2048), question feature (bs, 12, 100)

(1) question feature first fed into LSTM with `hidden_layer=16` and generated (bs, 1024) feature, then concat with image feature

(2) question feature first fed into LSTM with `hidden_layer=16` and generated (bs, 2048) feature, then element_mul with image feature

training epochs is set to 20

**Accuracy for (1)**

mini data test accuracy 100%, training data accuracy 46%, testing data 24%

**Accuracy for (2)

mini data test accuracy 80% 

