from mxnet.metric import EvalMetric, check_label_shapes
from mxnet import autograd
import mxnet.ndarray as nd
import mxnet as mx


class ZJLAccuracy(EvalMetric):
    def __init__(self, name='accuracy'):
        super(ZJLAccuracy, self).__init__(name)

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            if pred.shape != label.shape:
                pred = nd.argmax(pred, axis=1)
            pred = pred.asnumpy().astype('int32').flat
            label = label.asnumpy().astype('int32').flat

            check_label_shapes(label, pred)

            assert len(pred) % 3 == 0

            for i in range(0, len(pred), 3):
                if (pred[i] == label[i]) or (pred[i] == label[i+1]) or (pred[i] == label[i+2]):
                    self.sum_metric += 1
            self.num_inst += len(pred) / 3


def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    metric = ZJLAccuracy()
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        with autograd.record():
            data1 = batch.data[0].as_in_context(ctx)
            data2 = batch.data[1].as_in_context(ctx)
            data = [data1, data2]
            label = batch.label[0].as_in_context(ctx)
            output = net(data)

        metric.update([label], [output])
    return metric.get()[1]