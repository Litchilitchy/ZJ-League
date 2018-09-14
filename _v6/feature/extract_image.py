from mxnet import nd, image
import numpy as np
import os


from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
import logging


def load_image(img_path, long_side_length):
    x = image.imread(img_path)
    x = image.resize_short(x, long_side_length)
    x, _ = image.center_crop(x, (448, 448))
    x = x.astype('float32')
    x = x / 255
    x = image.color_normalize(x,
                              mean=nd.array([0.485, 0.456, 0.406]),
                              std=nd.array([0.229, 0.224, 0.225]))
    x = x.reshape((1, 3, 448, 448))

    return x


def get_image_feature(img_path):
    img_net = vision.inception_v3(pretrained=True)
    img = load_image(img_path, 448)

    feature = img_net.features(img)
    logging.debug("feature shape is %s", feature.shape)
    feature = feature.reshape(-1)

    return feature


def get_processed_state(data_path, is_test=False):
    file_list = os.listdir(data_path)
    idx = 0
    best_filename = None
    if is_test:
        pre_str = 'test'
    else:
        pre_str = 'train'

    for f in file_list:
        filename = f
        f = f.split('.')
        if len(f) != 2:
            continue
        if f[1] == 'npy':
            f = f[1].split('-')

            if f[1] == pre_str and len(f) == 2 and int(f[2]) > idx:
                idx = int(f[2])
                best_filename = filename

    if is_test:
        print('test feature idx start from', idx)
    else:
        print('train feature idx start from', idx)
    return idx, best_filename


def output_image_feature(data_path, is_test=False):
    image_feature = []
    file_list = os.listdir(data_path)
    file_list.sort()

    cnt = 0

    processed_img_num, processed_img_filename = get_processed_state(data_path, is_test)
    cur_img_idx = 0
    if processed_img_num != 0:
        image_feature = np.load(processed_img_filename).tolist()

    for filename in file_list:
        if cur_img_idx != processed_img_num:
            cur_img_idx += 1
            continue

        if filename.split('.')[1] != 'jpg':
            continue
        feature = get_image_feature(data_path + filename).asnumpy()
        image_feature.append(feature)

        cnt += 1
        if cnt % 100 == 0:
            ps = None
            if is_test:
                ps = 'test image'
            else:
                ps = 'train image'
            print('100 of %s is processed' % (ps))
        if cnt % 1000 == 0:
            tmp_nd = np.vstack(image_feature)
            if is_test:
                np.save('test-' + str(cnt) + '.npy', tmp_nd)
            else:
                np.save('train-' + str(cnt) + '.npy', tmp_nd)

    image_feature_nd = np.vstack(image_feature)
    print('check final shape', image_feature_nd.shape)
    if is_test:
        np.save('test_image.npy', image_feature_nd)
    else:
        np.save('train_image.npy', image_feature_nd)
    # output feature to file
    # feature shape (2048, )


output_image_feature('./../data/train_img/', False)
output_image_feature('./../data/test_img/', True)
