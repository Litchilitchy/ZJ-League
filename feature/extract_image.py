from mxnet import nd, image

from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn


def load_image(img_path, long_side_length):
    x = image.imread(img_path)
    x = image.resize_short(x, long_side_length)
    x, _ = image.center_crop(x, (224, 224))
    return x

def get_image_feature(img_path):
    img_net = vision.inception_v3(pretrained=True)
    img = load_image(img_path)

    feature = img_net.features(img)

    # now write the code to store the feature into file