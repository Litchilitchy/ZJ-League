import json
import os

import numpy as np

val_idx_cut = 3000
frame_per_video = 5

train_img = json.load(open('./../data/train_results.json'))
test_img = json.load(open('./../data/test_results.json'))
new_train_dict = {}
new_test_dict = {}
for d in train_img['results']:
    new_train_dict[d['img_name']] = np.array(d['scores'])
for d in test_img['results']:
    new_test_dict[d['img_name']] = np.array(d['scores'])


def output_attr(mode=None, vid_dict={}):
    if mode == 'train':
        v_idx = os.listdir('./../data/train_img')
        b, e = (0, val_idx_cut*frame_per_video)
    elif mode == 'val':
        v_idx = os.listdir('./../data/train_img')
        b, e = (val_idx_cut*frame_per_video, len(v_idx))
    elif mode == 'test':
        v_idx = os.listdir('./../data/test_img')
        b, e = (0, len(v_idx))
    else:
        assert mode == 'train' or mode == 'val' or mode == 'test'
    feature = []
    video_attr = []
    v_idx.sort()
    print('current mode ', mode, ' run in img range', b,e)
    for i in range(b, e):
        vid = v_idx[i]
        attr = vid_dict[vid]
        video_attr.append(attr)

        if (i+1)%frame_per_video == 0:
            feature.append(video_attr)
            video_attr = []
            tmp = np.array(feature)
            print(tmp.shape)

    nd = np.array(feature)
    print('final feature shape is', nd.shape)
    np.save(mode+'_attr.npy', nd)

print('new train and test dict size', len(new_train_dict), len(new_test_dict))

output_attr('train', new_train_dict)
output_attr('val', new_train_dict)
output_attr('test', new_test_dict)
