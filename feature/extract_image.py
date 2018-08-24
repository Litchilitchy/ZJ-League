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


def output_image_feature(data_path):
    image_feature = []
    file_list = os.listdir(data_path)
    file_list.sort()
    for filename in file_list:
        if filename.split('.')[1] != 'jpg':
            continue
        feature = get_image_feature('./../img/' + filename)
        image_feature.append(feature)

    np.save('image.npy', image_feature)
    # output feature to file
    # feature shape (2048, )


output_image_feature('./../img/')
