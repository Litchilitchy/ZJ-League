import numpy as np
import mxnet as mx


class DataIter(mx.io.DataIter):
    def __init__(self, image, question, answer, batch_size=20):
        self.idx = 0
        self.cur_idx = 0
        self.image = image
        self.question = question
        self.answer = answer

        self.provide_data = [('image', (batch_size, 2048)),
                             ('question', batch_size, 100)]
        self.provide_label = [('label', (batch_size, ))]

    def reset(self):
        self.idx = 0

    def next(self):
        if self.cur_idx == len(self.idx):
            raise StopIteration

        self.cur_idx += 1

        image = []
        question = []
        answer = []
        for video_frame_idx in range(3*self.cur_idx, 3*self.cur_idx+3):
            image.extend(self.image[video_frame_idx])
        image_cat = []
        for i in range(15):
            image_cat.append(image)
        for question_idx in range(5*self.cur_idx, 5*self.cur_idx+5):
            question.extend(self.question[question_idx])
            for answer_idx in range(3*question_idx, 3*question_idx+3):
                answer.extend(self.answer[answer_idx])

        data = [image_cat, question]
        return mx.io.DataBatch(data, [answer])

