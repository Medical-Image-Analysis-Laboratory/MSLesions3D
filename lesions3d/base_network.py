import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
from monai.networks.blocks import Convolution


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

config_no_maxpool = [
    # out_channel, stride, padding
    (32, (1, 1, 1), (1, 1, 1)),
    (32, (1, 1, 1), (1, 1, 1)),
    (64, (2, 2, 2), (1, 1, 1)),
    (64, (1, 1, 1), (1, 1, 1)),
    (128, (2, 2, 2), (1, 1, 1)),
    (128, (1, 1, 1), (1, 1, 1)),
    (256, (2, 2, 2), (1, 1, 1)),
    (256, (1, 1, 1), (1, 1, 1)),
]

config_maxpool_simple = [
    # out_channel, stride, padding
    (32, (1, 1, 1), (1, 1, 1)),
    (32, (1, 1, 1), (1, 1, 1)),
    ('maxpool3d', (2, 2, 2), (1, 1, 1)),
    (64, (1, 1, 1), (1, 1, 1)),
    ('maxpool3d', (2, 2, 2), (1, 1, 1)),
    (128, (1, 1, 1), (1, 1, 1)),
    ('maxpool3d', (2, 2, 2), (1, 1, 1)),
    (256, (1, 1, 1), (1, 1, 1)),
]

config_maxpool_double = [
    # out_channel, stride, padding
    (32, (1, 1, 1), (1, 1, 1)),
    (32, (1, 1, 1), (1, 1, 1)),
    ('maxpool3d', (2, 2, 2), (1, 1, 1)),
    (64, (1, 1, 1), (1, 1, 1)),
    (64, (1, 1, 1), (1, 1, 1)),
    ('maxpool3d', (2, 2, 2), (1, 1, 1)),
    (128, (1, 1, 1), (1, 1, 1)),
    (128, (1, 1, 1), (1, 1, 1)),
    ('maxpool3d', (2, 2, 2), (1, 1, 1)),
    (256, (1, 1, 1), (1, 1, 1)),
]


CONVNET_CONFIGS = {
    "convnet_strides": config_no_maxpool,
    "convnet_maxpool_simple": config_maxpool_simple,
    "convnet_maxpool_double": config_maxpool_double
}


class ConvNetBase(nn.Module):
    def __init__(self, aspect_ratios, config="convnet_maxpool_double", in_channels=1):
        # Aspect Ratio for maxpool = True: 4/7/10
        # Aspect Ratio for maxpool = False: ???
        super(ConvNetBase, self).__init__()

        self.config = CONVNET_CONFIGS[config]
        self.aspect_ratios = aspect_ratios
        self.in_channels = in_channels
        layers = []

        for i, (out_channels, stride, padding) in enumerate(self.config):
            if i > max(aspect_ratios.keys()):  # allows to truncate the network to desired shape based on aspect_ratios
                break
            if out_channels == 'maxpool3d':
                layers.append(torch.nn.MaxPool3d(kernel_size=(3, 3, 3),
                                                 stride=stride,
                                                 padding=padding))
            else:
                layers.append(Convolution(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    adn_ordering="NDA",
                    act=("prelu", {"init": 0.2}),
                    dropout=0.1,
                    strides=stride,
                    padding=padding
                ))
            in_channels = out_channels if type(out_channels) != str else in_channels

        self.features = torch.nn.Sequential(*layers)

    def init(self):
        for c in self.children():
            if isinstance(c, nn.Conv3d):
                nn.init.kaiming_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, image):
        out = image
        out_n_features = list(self.aspect_ratios.keys())
        out_features = {}
        for i, feat in enumerate(self.features):
            out = feat(out)
            # print(f"Layer {i} output  --  feature shape: {out.shape}")
            if i in out_n_features:
                out_features[i] = out
                if out.isnan().sum() > 0:
                    print("Yesssss this NaN error again in the base network")
                    raise Exception("Yesssss this NaN error again in the base network")

        return out_features

    def get_feature_map_infos(self, input_size, device):
        dummy_input = torch.randn((1, self.in_channels, *input_size)).to(device)
        feature_map_dimensions = {}
        features_n_channels = []
        for i, l in enumerate(self.features):
            dummy_input = l(dummy_input)
            feature_map_dimensions[i] = tuple(dummy_input.shape[-3:])
            features_n_channels.append(tuple(dummy_input.shape)[1])
        return feature_map_dimensions, features_n_channels
