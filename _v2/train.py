import numpy as np
import mxnet as mx
import logging
import json
from time import time
from data_iter import DataIter

import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd

import model


def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
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

    epochs = 1
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
                    if vid not in vid_to_ans.keys():
                        continue
                    tp = np.argsort(vid_to_ans[vid][i].asnumpy())[-3:].tolist()
                    for j in range(3):
                        if tp[j] > len(idx_to_ans):
                            tp[j] = str(0)
                        else:
                            tp[j] = str(tp[j])

                    ans += idx_to_ans[tp[0]] + ',' + idx_to_ans[tp[1]] + ',' + idx_to_ans[tp[2]]
                    if i != 4:
                        ans += ','
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
    log_file = open('time_log', 'w')
    ctx = mx.cpu()

    start_time = time()
    net = model.Net1()
    net.collect_params().initialize(mx.init.Xavier(), ctx)

    init_time = time()
    log_file.write('init time cost is ' + str(init_time-start_time) + '\n')

    train_img = np.load('feature/train_image.npy')
    train_ans = np.load('feature/train_answer.npy')
    train_q = np.load('feature/train_question.npy')

    test_img = np.load('feature/test_image.npy')
    print("Total test image:", test_img.shape[0])
    test_q = np.load('feature/test_question.npy')
    print("Total test question:", test_q.shape[0])

    data_train = DataIter(train_img, train_q, train_ans)

    load_time = time()
    log_file.write('data load time cost is ' + str(load_time - init_time) + '\n')
    #train(net, data_train, ctx)

    train_time = time()
    log_file.write('train time cost is ' + str(train_time - load_time) + '\n')
    ''' use following test data in official training step
    test_img = DataIter()
    test_q = DataIter()
    test_ans = DataIter()
    '''

    data_test = DataIter(test_img, test_q, train_ans)
    # vid_list is something get from video_idx_dict.json

    vid_dict = json.load(open('feature/video_idx_dict.json'))
    vid_list = []
    for k in vid_dict:
        vid_list.append(vid_dict[k])

    ans_idx = predict(net, data_test, ctx, vid_list)
    print('predict result shape is: ', len(ans_idx))
    # with open('_check_ans.json', 'w') as f:
    #    json.dump(ans_idx, f)

    predict_time = time()
    log_file.write('predict time cost is ' + str(predict_time - train_time) + '\n')
    log_file.close()
    ans_dict = json.load(open('feature/ans_dict.json'))

    output_data(ans_dict, ans_idx)

