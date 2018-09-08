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


def get_qa_pair_from_line(line, word_emd_dict={}, ans_dict={},
                          emb_dim=100):
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
            ans = 0
            ans_sentence = buff[j]

            if ans_sentence not in ans_dict.values():
                ans = len(ans_dict)
                ans_dict[ans] = ans_sentence
            else:
                ans = [k for k, v in ans_dict.items() if v == ans_sentence][0]
            qa_content.append(ans)

            '''ans_words = ans_sentence.strip('\n').split(' ')            
            for ans in ans_words:
                if ans not in word_emd_dict:
                    continue
                ans_vec += word_emd_dict[ans]
            qa_content.append(ans_vec)'''

    # [video_id, [q1], [a],[b],[c], [q2], [a],[b],[c] ]
    # [q][a][b][c] are 1-d vectors
    return qa_content


def output_question_feature(data_path, word_emd_dict={}, is_test=False):
    question_feature = []

    ans_dict = {}
    with open(data_path) as f:
        for line in f:
            feature = get_qa_pair_from_line(line, word_emd_dict, ans_dict)
            question_feature.append(feature)

    if not is_test:
        with open('ans_dict.json', 'w') as f:
            json.dump(ans_dict, f)

    question_feature.sort()

    video_idx_dict = {}

    for q in question_feature:
        video_idx_dict[len(video_idx_dict)] = q[0]

    if is_test:
        video_idx_dict_name = 'video_idx_dict.json'
        with open(video_idx_dict_name, 'w') as f:
            json.dump(video_idx_dict, f)
    else:
        video_idx_dict_name = 'video_idx_dict_train.json'

    q_list = []
    ans_list = []
    for question in question_feature:
        for i in range(1, len(question), 4):
            q_list.append(question[i])
            for j in range(i+1, i+4):
                ans_list.append(j)

    if is_test:
        np.save('test_question.npy', q_list)
    else:
        np.save('train_question.npy', np.array(q_list))
        np.save('train_answer.npy', ans_list)


word_dict = load_glove('./glove_model')
output_question_feature('./../data/train.txt', word_dict, False)
output_question_feature('./../data/test.txt', word_dict, True)
