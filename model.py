import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.gluon import nn
import numpy as np
import mxnet.ndarray.contrib as C
import gluonbook as gb

# the original easiest model v1 - simply concat
class Net1(gluon.Block):
    def __init__(self, num_category, **kwargs):
        super(Net1, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.bn = nn.BatchNorm()
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Dense(4096, activation="relu")
            self.fc2 = nn.Dense(num_category)
            self.image_lstm = gluon.rnn.LSTM(hidden_size=5)
            self.question_lstm = gluon.rnn.LSTM(hidden_size=12)
            self.image_fc = nn.Dense(1024, activation="relu")
            self.question_fc = nn.Dense(1024, activation="relu")
            self.ctx = gb.try_gpu()

    def forward(self, x):

        '''
        F = nd
        x1 = x[0]
        x1 = self.image_lstm(x1)
        x1 = self.image_fc(x1)
        x1 = nd.L2Normalization(x1)
        x2 = x[1]
        x2 = self.question_lstm(x2)
        x2 = self.question_fc(x2)
        x2 = nd.L2Normalization(x2)

        text_ones = F.ones((15, 1024), ctx = self.ctx)
        img_ones = F.ones((15, 1024), ctx = self.ctx)
        text_data = F.Concat(x1, text_ones,dim = 1)
        image_data = F.Concat(x2,img_ones,dim = 1)
        # Initialize hash tables
        S1 = F.array(np.random.randint(0, 2, (1,2048))*2-1,ctx = self.ctx)
        H1 = F.array(np.random.randint(0, 10000,(1,2048)),ctx = self.ctx)
        S2 = F.array(np.random.randint(0, 2, (1,2048))*2-1,ctx = self.ctx)
        H2 = F.array(np.random.randint(0, 10000,(1,2048)),ctx = self.ctx)
        # Count sketch
        cs1 = C.count_sketch( data = image_data, s=S1, h = H1 ,name='cs1',out_dim = 10000)
        cs2 = C.count_sketch( data = text_data, s=S2, h = H2 ,name='cs2',out_dim = 10000)
        fft1 = C.fft(data = cs1, name='fft1', compute_size = 15)
        fft2 = C.fft(data = cs2, name='fft2', compute_size = 15)
        c = fft1 * fft2
        ifft1 = C.ifft(data = c, name='ifft1', compute_size = 15)
        # MLP
        z = self.fc1(ifft1)
        z = self.bn(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z
        '''


        x1 = x[0]
        x1 = self.image_lstm(x1)
        x1 = self.image_fc(x1)
        x1 = nd.L2Normalization(x1)
        x2 = x[1]
        x2 = self.question_lstm(x2)
        x2 = self.question_fc(x2)
        x2 = nd.L2Normalization(x2)

        z = nd.elewise_mul(x1, x2)
        z = self.fc1(z)
        z = self.bn(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z
