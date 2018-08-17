import numpy as np
import mxnet as mx
import logging

import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd


def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    numerator = 0.
    denominator = 0.
    metric = mx.metric.Accuracy()
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


def train(net, data_train, data_eva, ctx=mx.cpu()):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    epochs = 10
    moving_loss = 0.
    best_eva = 0
    for e in range(epochs):
        data_train.reset()
        for i, batch in enumerate(data_train):
            data1 = batch.data[0].as_in_context(ctx)
            data2 = batch.data[1].as_in_context(ctx)
            data = [data1, data2]
            label = batch.label[0].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                cross_entropy = loss(output, label)
                cross_entropy.backward()
            trainer.step(data[0].shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            if i == 0:
                moving_loss = np.mean(cross_entropy.asnumpy()[0])
            else:
                moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
            # if i % 200 == 0:
            #    print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))
        eva_accuracy = evaluate_accuracy(data_eva, net)
        train_accuracy = evaluate_accuracy(data_train, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Eval_acc %s" % (e, moving_loss, train_accuracy, eva_accuracy))
        if eva_accuracy > best_eva:
            best_eva = eva_accuracy
            logging.info('Best validation acc found. Checkpointing...')
            net.save_params('vqa-mlp-%d.params' % (e))