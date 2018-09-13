import numpy as np
import mxnet as mx
import logging
import json
import os
from time import time
from data_iter import DataIter

import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd

from evaluate import evaluate_accuracy
import model


def load_params(net, data_path, ctx=mx.cpu()):
    file_list = os.listdir(data_path)
    best_params = None
    max_idx = -1
    for f in file_list:
        seg = f.split('.')
        if len(seg) != 2 or seg[1] != 'params':
            continue
        idx = seg[0].split('-')[2]
        if int(idx) > max_idx:
            max_idx = int(idx)
            best_params = f
    if best_params:
        net.load_params(best_params, ctx)
        print('best params file loaded', best_params)
    else:
        net.collect_params().initialize(mx.init.Xavier(), ctx)
    return max_idx+1


def train(net, data_train, start_epoch=0, ctx=mx.cpu()):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    epochs = 20
    moving_loss = 0.
    best_eva = 0
    for e in range(start_epoch, epochs):
        data_train.reset()
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
            moving_loss = np.mean(cross_entropy.asnumpy()[0])

            #if i == 0:
            #    moving_loss = np.mean(cross_entropy.asnumpy()[0])
            #else:
            #    moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])

            # if i % 200 == 0:
            #    print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))

        train_accuracy = evaluate_accuracy(data_train, net)
        print("Epoch %s. Loss: %s, Train_acc %s" % (e, moving_loss, train_accuracy))
        if train_accuracy > best_eva:
            best_eva = train_accuracy
            logging.info('Best validation acc found. Checkpointing...')
            net.save_params('vqa-mlp-%d.params' % (e))


def output_data(idx_to_ans={}, vid_to_ans={}):
    with open('data/test.txt', 'r') as f:

        with open('data/submit.txt', 'w') as out:
            for line in f:
                line = line.strip('\n').split(',')
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

                    tp = np.argmax(vid_to_ans[vid][i*3].asnumpy())

                    if str(tp) not in idx_to_ans:
                        tp = '0'
                    ans += idx_to_ans[str(tp)]
                    if i != 4:
                        ans += ','
                ans += '\n'
                out.write(ans)
    return


def predict(net, data_test, ctx=mx.cpu()):
    # vid_list store the order of data_test, string type
    # vid_list = [vid1, vid2, vid3, ...]
    vid_dict = json.load(open('feature/video_idx_dict.json'))
    vid_list = []
    for k in vid_dict:
        vid_list.append(vid_dict[k])

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


if __name__ == '__main__':
    log_file = open('time_log', 'w')
    ctx = mx.cpu()

    start_time = time()

    ans_dict = json.load(open('feature/ans_dict.json'))
    num_category = len(ans_dict)
    net = model.Net1(num_category)

    start_epoch = load_params(net, './', ctx)

    init_time = time()
    log_file.write('init time cost is ' + str(init_time-start_time) + '\n')

    train_img = np.load('feature/train_image.npy')
    train_ans = np.load('feature/train_answer.npy')
    train_q = np.load('feature/train_question.npy')

    test_img = np.load('feature/test_image.npy')
    print("Total test image:", test_img.shape[0])
    test_q = np.load('feature/test_question.npy')
    print("Total test question:", test_q.shape[0])

    data_train = DataIter(train_img, train_q, train_ans, False)

    load_time = time()
    log_file.write('data load time cost is ' + str(load_time - init_time) + '\n')
    train(net, data_train, start_epoch=start_epoch, ctx=ctx)

    train_time = time()
    log_file.write('train time cost is ' + str(train_time - load_time) + '\n')

    data_test = DataIter(test_img, test_q, train_ans, True)

    ans_idx = predict(net, data_test, ctx)
    print('predict result shape is: ', len(ans_idx))

    predict_time = time()
    log_file.write('predict time cost is ' + str(predict_time - train_time) + '\n')
    log_file.close()


    output_data(ans_dict, ans_idx)

