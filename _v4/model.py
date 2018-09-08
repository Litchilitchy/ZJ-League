import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.gluon import nn


# the original easiest model v1 - simply concat
class Net1(gluon.Block):
    def __init__(self, num_category, **kwargs):
        super(Net1, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.bn = nn.BatchNorm()
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Dense(8192, activation="relu")
            self.fc2 = nn.Dense(num_category)

            self.lstm = gluon.rnn.LSTM(hidden_size=16)
            self.lstm_fc = nn.Dense(1024, activation="relu")

    def forward(self, x):
        x1 = nd.L2Normalization(x[0])
        x2 = x[1]
        x2 = self.lstm(x2)
        x2 = self.lstm_fc(x2)
        x2 = nd.L2Normalization(x2)

        z = nd.concat(x1, x2, dim=1)
        z = self.fc1(z)
        z = self.bn(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z
