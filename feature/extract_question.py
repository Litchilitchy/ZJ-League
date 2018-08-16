from mxnet.gluon import nn

import urllib.request
import sys
import os
import zipfile
import numpy as np
import logging
import pickle
import time


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove(data_dir_path, to_file_path):
    if not os.path.exists(to_file_path):
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)

        glove_zip = data_dir_path + '/glove.6B.zip'

        if not os.path.exists(glove_zip):
            logging.debug('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        logging.debug('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall(data_dir_path)
        zip_ref.close()


def load_glove(data_dir_path=None, embedding_dim=None):
    """
    Load the glove models (and download the glove model if they don't exist in the data_dir_path
    :param data_dir_path: the directory path on which the glove model files will be downloaded and store
    :param embedding_dim: the dimension of the word embedding, available dimensions are 50, 100, 200, 300, default is 100
    :return: the glove word embeddings
    """
    if embedding_dim is None:
        embedding_dim = 100

    glove_pickle_path = data_dir_path + "/glove.6B." + str(embedding_dim) + "d.pickle"
    if os.path.exists(glove_pickle_path):
        logging.info('loading glove embedding from %s', glove_pickle_path)
        start_time = time.time()
        with open(glove_pickle_path, 'rb') as handle:
            result = pickle.load(handle)
            duration = time.time() - start_time
            logging.debug('loading glove from pickle tooks %.1f seconds', (duration ))
            return result
    glove_file_path = data_dir_path + "/glove.6B." + str(embedding_dim) + "d.txt"
    download_glove(data_dir_path, glove_file_path)
    _word2em = {}
    logging.debug('loading glove embedding from %s', glove_file_path)
    file = open(glove_file_path, mode='rt', encoding='utf8')
    for i, line in enumerate(file):
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
        if i % 1000 == 0:
            logging.debug('loaded %d %d-dim glove words', i, embedding_dim)
    file.close()
    with open(glove_pickle_path, 'wb') as handle:
        logging.debug('saving glove embedding as %s', glove_pickle_path)
        pickle.dump(_word2em, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return _word2em


def get_qa_pair_from_line(line):
    buff = line.split(',')

    if (len(buff) - 1) % 4 != 0:
        raise AssertionError("qa length error, not times of 4")
    pair = []
    for i in range(1, len(buff), 4):
        for j in range(i+1, i+4):
            pair.append([buff[i], buff[j]])

    # [ [q, ans1], [q, ans2], [q, ans3] ]
    return pair
