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


def output_data(idx_to_ans={}, vid_to_ans={}):
    with open('submit.txt', 'r') as f:

        with open('my_submit.txt', 'w') as out:

            line = f.readline().split(',')
            assert len(line) == 21
            vid = line[0]
            q = []
            for i in range(1, 21, 4):
                q.append(line[i])

            ans = ''
            ans += vid
            ans += ','
            for i in range(5):
                ans += q[i]
                ans += ','
                tp = np.argsort(vid_to_ans[vid][i].asnumpy())[-3:]
                for j in range(3):
                    if tp[j] > len(idx_to_ans):
                        tp[j] = str(0)

                ans += idx_to_ans[tp[0]] + ',' + idx_to_ans[tp[1]] + ',' + idx_to_ans[tp[2]]
            ans += '\n'
            out.write(ans)
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
            '''
            if vid_list[i] not in ans:
                ans[vid_list[i]] = []
            ans[vid_list[i]].append(output)'''
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
    # vid_list is something get from video_idx_dict.json
    vid_list = ['ZJL963', 'ZJL2495', 'ZJL3540']
    ans_idx = predict(net, data_test, ctx, vid_list)

    ans_dict = json.load(open('feature/ans_dict.json'))

    output_data(ans_dict, ans_idx)

