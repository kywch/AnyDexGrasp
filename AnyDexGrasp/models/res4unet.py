import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiReLU
import MinkowskiEngine.MinkowskiOps as me

from .resnet import ResNetBase, get_norm
from .common import ConvType, NormType, conv, conv_tr
from .resnet_block import BasicBlock


class Res4UNetBase(ResNetBase):
    BLOCK = None
    # PLANES = (64, 128, 256, 128, 128)
    PLANES = (32, 64, 128, 64, 64)  # upd.
    DILATIONS = (1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2)
    INIT_DIM = PLANES[0]
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    # Upd Note. ME v0.5 delete "ME.RegionType.HYBRID".
    # CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels, out_channels, conv1_kernel_size, bn_momentum, D=3, **kwargs):
        super(Res4UNetBase, self).__init__(in_channels, out_channels, conv1_kernel_size, self.DILATIONS, bn_momentum, D)

    def network_initialization(self, in_channels, out_channels, conv1_kernel_size, dilations, bn_momentum, D):
        # Setup net_metadata
        bn_momentum = bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv1p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn1 = get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr3p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[3],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr3 = get_norm(self.NORM_TYPE, self.PLANES[3], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[3] + self.PLANES[1] * self.BLOCK.expansion
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr4p2s1 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.relu = MinkowskiReLU(inplace=True)

        self.feature_extraction = nn.Sequential(
            conv(
                self.PLANES[4] + self.PLANES[0] * self.BLOCK.expansion,
                128,  # 256,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=False,
                D=D,
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        # D=D), ME.MinkowskiBatchNorm(256), ME.MinkowskiReLU())
        self.final = conv(128, out_channels, kernel_size=1, stride=1, dilation=1, bias=True, D=D)
        # self.final = conv(256, out_channels, kernel_size=1, stride=1, dilation=1, bias=True, D=D)

    def forward(self, x, return_features=False):
        out = self.conv1p1s1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_b1p1 = self.block1(out)

        out = self.conv2p1s2(out_b1p1)
        out = self.bn2(out)
        out = self.relu(out)

        out_b2p2 = self.block2(out)

        out = self.conv3p2s2(out_b2p2)
        out = self.bn3(out)
        out = self.relu(out)

        # pixel_dist=4
        out = self.block3(out)

        out = self.convtr3p4s2(out)
        out = self.bntr3(out)
        out = self.relu(out)

        # pixel_dist=2
        out = me.cat(out, out_b2p2)
        out = self.block4(out)

        out = self.convtr4p2s1(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # pixel_dist=1
        out = me.cat(out, out_b1p1)
        features = self.feature_extraction(out)
        out = self.final(features)

        if return_features:
            return out, features
        else:
            return out


class Res4UNet14(Res4UNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)
