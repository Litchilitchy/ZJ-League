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

import model


def load_params(net, data_path, ctx=mx.cpu()):
    file_list = os.listdir(data_path)
    best_params = None
    max_idx = 0
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
    return max_idx


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

    assert len(data_test.image) == len(vid_list)
    ans = {}

    check_file = open('check_test_data.txt','w')
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
    check_file.close()
    return ans


if __name__ == '__main__':

    ctx = mx.cpu()

    start_time = time()

    ans_dict = json.load(open('feature/ans_dict.json'))
    num_category = len(ans_dict)
    net = model.Net1(num_category)

    start_epoch = load_params(net, './', ctx)

    test_img = np.load('feature/test_image.npy')
    print("Total test image:", test_img.shape[0])
    test_q = np.load('feature/test_question.npy')
    print("Total test question:", test_q.shape[0])

    data_test = DataIter(test_img, test_q, np.zeros(test_img.shape[0]*15))

    ans_idx = predict(net, data_test, ctx)
    print('predict result shape is: ', len(ans_idx))

    predict_time = time()

    output_data(ans_dict, ans_idx)

