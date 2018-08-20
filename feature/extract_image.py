from mxnet import nd, image
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
    # 这里一行需要改，由(448,448,3) 到 (3,448,448)不能用reshape而应该用transform，
    # 并且可以不用前面的那个1，所有图片拼接成batch就是4-D的了（输入要求是4-D (batch_size, channel, W, H)）
    # 因为之前要对单个图片测试，要做成一个batch，所以才弄成了这样

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
    for filename in os.listdir(data_path):
        feature = get_image_feature(os.path.join(filename))
        image_feature.append(feature)

    # output feature to file
    # 此处需要先看一下feature的形状，image_feature是feature的简单list集合
