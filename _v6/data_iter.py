import numpy as np
import mxnet as mx
import mxnet.ndarray as nd


class DataIter(mx.io.DataIter):
    def __init__(self, image, question, answer, is_test):
        print('image sample', len(image), 'question sample', len(question),'answer sample', len(answer))
        assert len(image)*5 == len(question)
        if not is_test:
            assert len(image)*15 == len(answer)

        self.idx = []
        self.cur_idx = 0
        self.image = image
        self.question = question
        self.answer = answer

        for i in range(int(len(self.image))):
            self.idx.append(i)

        self.provide_data = [('image', (15, 2048)),
                             ('question', 15, 100)]
        self.provide_label = [('label', (15, ))]

    def reset(self):
        self.cur_idx = 0

    def next(self):

        if self.cur_idx >= len(self.idx):
            raise StopIteration

        question = []
        answer = []

        image = self.image[self.cur_idx]
        image_batch = []
        for i in range(15):
            image_batch.append(image)

        for question_idx in range(5*self.cur_idx, 5*self.cur_idx+5):

            for answer_idx in range(3*question_idx, 3*question_idx+3):
                question.append(self.question[question_idx])
                answer.append(self.answer[answer_idx])

        image_batch = nd.array(image_batch)
        question = nd.array(question)
        answer = nd.array(answer)
        data = [image_batch, question]

        self.cur_idx += 1

        return mx.io.DataBatch(data, [answer])

