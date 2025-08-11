import math
import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


###########################################################


# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(
        self,
        in_channels=9,
        dim_hidden=64 * 26,
        num_classes=6,
        conv_kernel_size=(1, 9),
        pool_kernel_size=(1, 2),
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/digit5.py
class Digit5CNN(nn.Module):
    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module(
            "conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        )
        self.encoder.add_module("bn1", nn.BatchNorm2d(64))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module(
            "maxpool1",
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.encoder.add_module(
            "conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        )
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module(
            "maxpool2",
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.encoder.add_module(
            "conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        )
        self.encoder.add_module("bn3", nn.BatchNorm2d(128))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("bn4", nn.BatchNorm1d(3072))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("bn5", nn.BatchNorm1d(2048))
        self.linear.add_module("relu5", nn.ReLU())

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            # nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out


# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.fc(x)
#         return x


import torch
import torch.nn as nn

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=2): # Removed 'dim' as it's no longer needed
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        # Use nn.LazyLinear to automatically determine the input dimension
        self.fc1 = nn.Sequential(
            nn.LazyLinear(512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


# ====================================================================================================================


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# ====================================================================================================================


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


# ====================================================================================================================


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


# ====================================================================================================================


class DNN(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================


class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LeNet(nn.Module):
    def __init__(
        self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_classes=10, iswn=None
    ):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================


class LSTMNet(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers=2,
        bidirectional=False,
        dropout=0.2,
        padding_idx=0,
        vocab_size=98635,
        num_classes=10,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]

        embedded = self.embedding(text)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out


# ====================================================================================================================


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out


# ====================================================================================================================


class TextCNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_channels=100,
        kernel_size=[3, 4, 5],
        max_len=200,
        dropout=0.8,
        padding_idx=0,
        vocab_size=98635,
        num_classes=10,
    ):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=num_channels,
                kernel_size=kernel_size[0],
            ),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=num_channels,
                kernel_size=kernel_size[1],
            ),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=num_channels,
                kernel_size=kernel_size[2],
            ),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1),
        )

        self.dropout = nn.Dropout(dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out


# ====================================================================================================================


# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input

#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output


# ====================================================================================================================


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape

    h_out = (
        np.floor(
            (n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1))
            / stride[0]
        ).astype(int)
        + 1
    )
    w_out = (
        np.floor(
            (m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1))
            / stride[1]
        ).astype(int)
        + 1
    )
    b = [kernel_side // 2, kernel_side // 2]
    return h_out, w_out, batch_size, n_channels


def multiple_convs_kan_conv2d(
    matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
    kernels,
    kernel_side,
    out_channels,
    stride=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    device="cuda",
) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(
        matrix, kernel_side, stride, dilation, padding
    )
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out)).to(
        device
    )  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold(
        (kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride
    )
    conv_groups = (
        unfold(matrix[:, :, :, :])
        .view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out)
        .transpose(2, 3)
    )  # reshape((batch_size,n_channels,h_out,w_out))
    # for channel in range(n_channels):
    kern_per_out = len(kernels) // out_channels
    # print(len(kernels),out_channels)
    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)

        # Aggregate outputs from each kernel assigned to this output channel
        for k_idx in range(kern_per_out):
            kernel = kernels[c_out * kern_per_out + k_idx]
            conv_result = kernel.conv.forward(
                conv_groups[:, k_idx, :, :].flatten(0, 1)
            )  # Apply kernel with non-linear function
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum  # Store results in output tensor

    return matrix_out


def add_padding(matrix: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix.

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix

    return padded_matrix


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=True):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
        device: str = "cpu",
    ):
        """
        Kan Convolutional Layer with multiple convolutions

        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_spline (float): Scale of the spline
            base_activation (torch.nn.Module): Activation function
            grid_eps (float): Epsilon of the grid
            grid_range (tuple): Range of the grid
            device (str): Device to use
        """

        super(KAN_Convolutional_Layer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        # self.device = device
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.stride = stride

        # Create n_convs KAN_Convolution objects
        for _ in range(in_channels * out_channels):
            self.convs.append(
                KAN_Convolution(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    # device = device ## changed device to be allocated as per the input device for pytorch DDP
                )
            )

    def forward(self, x: torch.Tensor):
        # If there are multiple convolutions, apply them all
        self.device = x.device
        # if self.n_convs>1:
        return multiple_convs_kan_conv2d(
            x,
            self.convs,
            self.kernel_size[0],
            self.out_channels,
            self.stride,
            self.dilation,
            self.padding,
            self.device,
        )

        # If there is only one convolution, apply it
        # return self.convs[0].forward(x)


class KAN_Convolution(torch.nn.Module):
    def __init__(
        self,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
        device="cpu",
    ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.device = device
        self.conv = KANLinear(
            in_features=math.prod(kernel_size),
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x: torch.Tensor):
        self.device = x.device
        return multiple_convs_kan_conv2d(
            x,
            [self],
            self.kernel_size[0],
            1,
            self.stride,
            self.dilation,
            self.padding,
            self.device,
        )

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class KANC_MLP_Medium(nn.Module):
    def __init__(self, num_classes=2, grid_size: int = 5, **kwargs):
        super().__init__()
        
        # Layer definitions from the provided architecture
        self.conv1 = KAN_Convolutional_Layer(in_channels=1, out_channels=5, kernel_size=(3,3), grid_size=grid_size, padding=(1,1))
        self.conv2 = KAN_Convolutional_Layer(in_channels=5, out_channels=10, kernel_size=(3,3), grid_size=grid_size, padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        
        # The fc layer is corrected for a 128x128 input image
        # After two 2x2 pooling layers, a 128x128 image becomes 32x32.
        # The number of features is out_channels * height * width = 10 * 32 * 32 = 10240.
        self.fc = nn.Linear(10240, num_classes)
        
        self.name = f"KANC_MLP_Medium (gs = {grid_size})"

    def forward(self, x):
        # Input: (batch, 1, 128, 128)
        x = self.conv1(x)
        x = self.pool1(x) # -> (batch, 5, 64, 64)

        x = self.conv2(x)
        x = self.pool1(x) # -> (batch, 10, 32, 32)
        
        x = self.flat(x)
        x = self.fc(x)
        
        # Return raw logits, as nn.CrossEntropyLoss expects this
        return x
    
# ====================================================================================================================

class DeepConvKAN(nn.Module):
    """
    This "Deep and Narrow" architecture is designed to prevent out-of-memory errors
    on 128x128 images without using adaptive pooling. It adds more pooling layers
    to progressively reduce the spatial dimensions, resulting in a much smaller
    feature vector before the classifier.
    """
    def __init__(self, num_classes=2, grid_size: int = 5, **kwargs):
        super().__init__()

        self.body = nn.Sequential(
            # Block 1
            KAN_Convolutional_Layer(in_channels=1, out_channels=4, kernel_size=(3,3), grid_size=grid_size, padding=(1,1)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x128 -> 64x64

            # Block 2
            KAN_Convolutional_Layer(in_channels=4, out_channels=8, kernel_size=(3,3), grid_size=grid_size, padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64 -> 32x32

            # Block 3
            KAN_Convolutional_Layer(in_channels=8, out_channels=16, kernel_size=(3,3), grid_size=grid_size, padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            # Block 4
            KAN_Convolutional_Layer(in_channels=16, out_channels=32, kernel_size=(3,3), grid_size=grid_size, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )
        
        # Classifier Head
        # The flattened feature vector size is now manageable: 32 channels * 8 * 8 = 2048
        fc_input_features = 32 * 8 * 8
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5), # Add dropout for regularization
            KANLinear(in_features=fc_input_features, out_features=32, grid_size=grid_size),
            KANLinear(in_features=32, out_features=num_classes, grid_size=grid_size)
        )
        
        self.name = f"Fed-Conv-KAN_DeepNarrow (gs = {grid_size})"

    def forward(self, x):
        x = self.body(x)
        x = self.fc(x)
        return x


# ====================================================================================================================


class ResidualBlock(nn.Module):
    """
    A building block for a ResNet, containing two convolutional layers
    and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut (skip connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, use a 1x1 convolution to match them
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # The output of the main path is added to the (transformed) input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FedLightCNN(nn.Module):
    """
    A state-of-the-art, lightweight ResNet-style architecture designed for
    high performance on medical image classification tasks like the
    ChestXRay dataset.
    """
    def __init__(self, num_classes=2, **kwargs):
        super(FedLightCNN, self).__init__()
        self.in_channels = 16

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Stacking residual blocks to build a deep network
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        self.name = "Fed-ResNet-CNN"

    def _make_layer(self, out_channels, num_blocks, stride):
        """Helper function to create a series of residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.adaptive_pool(out)
        # Flatten the output for the final linear layer
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

# ====================================================================================================================


class ConvKAN(nn.Module):
    """
    This is the first modification of the original paper's architecture,
    designed to reduce memory usage. The convolutional body matches the paper,
    but the classifier head is made smaller.
    """
    def __init__(self, num_classes=2, grid_size: int = 5, **kwargs):
        super().__init__()
        
        # --- Feature Extractor Body (as per paper) ---
        
        # Using the paper's original channel sizes
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=1, 
            out_channels=16, 
            kernel_size=(3,3), 
            grid_size=grid_size, 
            padding=(1,1)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = KAN_Convolutional_Layer(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(3,3), 
            grid_size=grid_size, 
            padding=(1,1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.flat = nn.Flatten()
        
        # --- KAN-based Classifier Head (MODIFIED) ---
        
        # The flattened feature vector size is very large: 32 * 32 * 32 = 32768
        fc_input_features = 32 * 32 * 32
        
        self.fc = nn.Sequential(
            # MODIFICATION: Reduced hidden units from 64 to 16
            KANLinear(in_features=fc_input_features, out_features=16, grid_size=grid_size),
            
            # The second KAN layer now takes 16 features instead of 64
            KANLinear(in_features=16, out_features=num_classes, grid_size=grid_size)
        )
        
        self.name = f"Fed-Conv-KAN_Paper_Mod1 (gs = {grid_size})"

    def forward(self, x):
        # Input: (batch, 1, 128, 128)
        
        # Feature Extractor
        x = self.conv1(x) # -> (batch, 16, 128, 128)
        x = self.pool1(x) # -> (batch, 16, 64, 64)

        x = self.conv2(x) # -> (batch, 32, 64, 64)
        x = self.pool2(x) # -> (batch, 32, 32, 32)
        
        # Classifier
        x = self.flat(x)
        x = self.fc(x)
        
        # Return raw logits
        return x
