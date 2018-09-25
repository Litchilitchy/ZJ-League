from mxnet.gluon import nn

import urllib.request
import sys
import os
import zipfile
import numpy as np
import logging
import pickle
import time
import json


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
    :param embedding_dim: t，he dimension of the word embedding, available dimensions are 50, 100, 200, 300, default is 100
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


def get_qa_pair_from_line(line, word_emd_dict={}, ans_dict={}):
    buff = line.strip('\n').split(',')

    if (len(buff) - 1) % 4 != 0:
        raise AssertionError("qa length error, not times of 4")
    if (len(buff) - 1) / 4 != 5:
        raise AssertionError("qa length error, not 5 questions")

    qa_content = []
    qa_content.append(buff[0])

    # every questions has 3 answers, 3+1 = 4
    for i in range(1, len(buff), 4):
        # question
        q_sentence = buff[i]
        q_words = q_sentence.strip('\n').split(' ')
        q_vec = []
        '''
        for q in q_words:
            if q not in word_emd_dict:
                continue
            q_vec.append(word_emd_dict[q])
        '''
        cur_len = 0
        j = cur_len
        while cur_len != 12:
            if j < len(q_words):
                cur_word = q_words[j]
                if cur_word not in word_emd_dict:
                    if len(cur_word.split("'")) > 1:
                        if cur_word.split("'")[0] not in word_emd_dict:
                            j += 1
                            continue
                        q_vec.append(word_emd_dict[cur_word.split("'")[0]])
                        cur_len += 1
                        j += 1
                    elif len(cur_word.split('-')) > 1:
                        q_vec.append(word_emd_dict[cur_word.split('-')[0]])
                        q_vec.append(word_emd_dict[cur_word.split('-')[1]])
                        cur_len += 2
                        j += 1
                        assert cur_len <= 12
                    else:
                        j += 1
                        continue
                else:
                    q_vec.append(word_emd_dict[cur_word])
                    cur_len += 1
                    j += 1
            else:
                q_vec.append(np.zeros(100))
                cur_len += 1

        qa_content.append(q_vec)

        # for the 3 questions, i+1 to i+3
        # ans1, ans2, ans3
        for j in range(i+1, i+4):
            ans_sentence = buff[j]

            if ans_sentence not in ans_dict.values():
                ans = len(ans_dict)
                ans_dict[ans] = ans_sentence
            else:
                ans = [k for k, v in ans_dict.items() if v == ans_sentence][0]
            qa_content.append(ans)

    # [video_id, [q1], [a],[b],[c], [q2], [a],[b],[c] ]
    # [q][a][b][c] are 1-d vectors
    return qa_content


data_files = {'train': './../data/train.txt',
              'val': './../data/val.txt',
              'test': './../data/test.txt'}
output_files = {'train': ('train_question.npy', 'train_answer.npy'),
                'val': ('val_question.npy', 'val_answer.npy'),
                'test': ('test_question.npy', 'video_idx_dict.json', 'ans_dict.json')}
train_data, val_data, test_data = data_files
test_q, v_idx, ans_idx = output_files['test']


def get_feature_from_data(mode=None, val_cut_idx=0, word_emd_dict={}, ans_dict={}):
    question_feature = []
    with open(data_files[mode]) as f:
        for line in f:
            feature = get_qa_pair_from_line(line, word_emd_dict, ans_dict)
            question_feature.append(feature)

    question_feature.sort()
    if mode == 'train':
        output_feature_to_npy(question_feature[:val_cut_idx], 'train')
        output_feature_to_npy(question_feature[val_cut_idx:], 'val')
    else:
        output_feature_to_npy(question_feature, mode)
    return


def output_feature_to_npy(feature=[], mode=None):
    assert len(feature) != 0
    feature.sort()

    q_list = []
    ans_list = []
    for question in feature:
        for i in range(1, len(question), 4):
            q_list.append(question[i])
            for j in range(i+1, i+4):
                ans_list.append(question[j])

    if mode == 'test':
        video_idx_dict = {}
        for q in feature:
            video_idx_dict[len(video_idx_dict)] = q[0]
        with open(v_idx, 'w') as f:
            json.dump(video_idx_dict, f)
        np.save(test_q, q_list)
    else:
        np.save(output_files[mode][0], q_list)
        np.save(output_files[mode][1], ans_list)


def output_question_feature(word_emd_dict={}, val_cut_idx=0):
    ans_dict = {}
    get_feature_from_data('train', val_cut_idx=val_cut_idx,
                          word_emd_dict=word_emd_dict, ans_dict=ans_dict)
    with open(ans_idx, 'w') as f:
        json.dump(ans_dict, f)

    get_feature_from_data('test', val_cut_idx=val_cut_idx,
                          word_emd_dict=word_emd_dict, ans_dict=ans_dict)


word_dict = load_glove('./glove_model')
output_question_feature(word_emd_dict=word_dict, val_cut_idx=3000)

