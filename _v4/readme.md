##### v4 模型

change question feature to 3-d array, with `layout=NTC`

modify the network structure, use lstm with `hidden_layer=16` to process the 3-d array to 2-d array, with each element of the array is a `shape=(1024,)` feature vector

**Accuracy:**

mini data test accuracy 70% after 10 epochs (not converge and continue to be 97% after 30 epochs), training data accuracy 99.97% after 10 epochs
