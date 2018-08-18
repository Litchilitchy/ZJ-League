from mxnet import nd, image
import os


from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
import logging


def load_image(img_path, long_side_length):
    x = image.imread(img_path)
    x = image.resize_short(x, long_side_length)
    x, _ = image.center_crop(x, (224, 224))
    return x


def get_image_feature(img_path):
    img_net = vision.inception_v3(pretrained=True)
    img = load_image(img_path)

    feature = img_net.features(img)
    logging.debug("feature shape is %s", nd.shape(feature))

    return feature


def output_image_feature(data_path):
    image_feature = []
    for filename in os.listdir(data_path):
        feature = get_image_feature(os.path.join(filename))
        image_feature.append(feature)
    
    # output feature to file
