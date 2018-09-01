import numpy as np
import mxnet as mx
import logging
import json
from data_iter import DataIter

import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd

import model


def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    numerator = 0.
    denominator = 0.
    metric = mx.metric.Accuracy()
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        with autograd.record():
            data1 = batch.data[0].as_in_context(ctx)
            data2 = batch.data[1].as_in_context(ctx)
            data = [data1, data2]
            label = batch.label[0].as_in_context(ctx)
            output = net(data)

        metric.update([label], [output])
    return metric.get()[1]


def train(net, data_train, ctx=mx.cpu()):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    epochs = 10
    moving_loss = 0.
    best_eva = 0
    for e in range(epochs):

        for i, batch in enumerate(data_train):
            data1 = batch.data[0].as_in_context(ctx)
            data2 = batch.data[1].as_in_context(ctx)
            data = [data1, data2]
            label = batch.label[0].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                cross_entropy = loss(output, label)
                cross_entropy.backward()
            trainer.step(data[0].shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            if i == 0:
                moving_loss = np.mean(cross_entropy.asnumpy()[0])
            else:
                moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
            # if i % 200 == 0:
            #    print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))

        train_accuracy = evaluate_accuracy(data_train, net)
        print("Epoch %s. Loss: %s, Train_acc %s" % (e, moving_loss, train_accuracy))
        if train_accuracy > best_eva:
            best_eva = train_accuracy
            logging.info('Best validation acc found. Checkpointing...')
            net.save_params('vqa-mlp-%d.params' % (e))


def output_data(ans=[[]], map_index_to_vid={}):
    with open('submit.txt', 'w') as f:
        for i in range(len(map_index_to_vid)):
            line = ""
            line += map_index_to_vid[i] + ','
            for j in range(5):

                if i != 4:
                    line += ','
            line += '\n'
            f.write(line)
    return


def predict(net, data_test, ctx=mx.cpu(), vid_list=[]):
    # vid_list store the order of data_test, string type
    # vid_list = [vid1, vid2, vid3, ...]
    ans = {}
    for i, batch in enumerate(data_test):
        with autograd.record():
            image = batch.data[0].as_in_context(ctx)
            question = batch.data[1].as_in_context(ctx)
            data = [image, question]
            # label = batch.label[0].as_in_context(ctx)
            # label_one_hot = nd.one_hot(label, 10)
            output = net(data)
            ans[vid_list[i]] = output

    return ans
    # output = np.argmax(output.asnumpy(), axis=1)


if __name__ == '__main__':
    ctx = mx.cpu()
    net = model.Net1()
    net.collect_params().initialize(mx.init.Xavier(), ctx)
    train_img = np.load('feature/image.npy')
    train_ans = np.load('feature/answer.npy')
    train_q = np.load('feature/question.npy')

    data_train = DataIter(train_img, train_q, train_ans)
    train(net, data_train, ctx)

    ''' use following test data in official training step
    test_img = DataIter()
    test_q = DataIter()
    test_ans = DataIter()
    '''

    data_test = DataIter(train_img, train_q, train_ans)
    ans_idx = predict(net, data_test, ctx)

    ans_dict = json.load('feature/ans_dict.json')
    ans = ans_dict[ans_idx]
    output_data(ans)

