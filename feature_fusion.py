import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from feature_extraction import ResNet50

# backbone nets
#backbone_nets = {'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19}
backbone_nets = {'ResNet50': ResNet50}


# aggregation
class AvgFeatAGG2d(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(self, kernel_size, output_size=None, dilation=1, stride=1, device=torch.device('cpu')):
        super(AvgFeatAGG2d, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size

    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def forward(self, input):
        N, C, H, W = input.shape
        output = self.unfold(input)  # (b, cxkxk, h*w)
        output = torch.reshape(output, (N, C, int(self.kernel_size[0]*self.kernel_size[1]), int(self.output_size[0]*self.output_size[1])))
        # print(output.shape)
        output = torch.mean(output, dim=2)
        # output = self.fold(input)
        return output


class featureFusion(nn.Module):
    r"""
    Build muti-scale regional feature based on ResNet50-feature maps.
    """

    def __init__(self, backbone='ResNet50',
                 cnn_layers=("relu1_1",),
                 upsample="nearest",
                 is_agg=True,
                 kernel_size=(4, 4),
                 stride=(4, 4),
                 dilation=1,
                 featmap_size=(128, 128),
                 device='cpu'):

        super(featureFusion, self).__init__()
        self.device = torch.device(device)
        self.feature = backbone_nets[backbone]()    # build backbone net
        self.feat_layers = cnn_layers
        self.is_agg = is_agg
        self.map_size = featmap_size
        self.upsample = upsample
        self.patch_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # feature processing
        padding_h = (self.patch_size[0] - self.stride[0]) // 2
        padding_w = (self.patch_size[1] - self.stride[1]) // 2
        self.padding = (padding_h, padding_w)
        self.replicationpad = nn.ReplicationPad2d((padding_w, padding_w, padding_h, padding_h))

        self.out_h = int((self.map_size[0] + 2*self.padding[0] - (self.dilation * (self.patch_size[0] - 1) + 1)) / self.stride[0] + 1)
        self.out_w = int((self.map_size[1] + 2*self.padding[1] - (self.dilation * (self.patch_size[1] - 1) + 1)) / self.stride[1] + 1)
        self.out_size = (self.out_h, self.out_w)
        # print(self.out_size)
        self.feat_agg = AvgFeatAGG2d(kernel_size=self.patch_size, output_size=self.out_size,
                                    dilation=self.dilation, stride=self.stride, device=self.device)
        self.unfold = nn.Unfold(kernel_size=1, dilation=1, padding=0, stride=1)
        # self.features = torch.Tensor()

    def forward(self, input):
        feat_maps = self.feature(input)
        features = torch.Tensor().to(self.device)
        # extracting features
        for _, feat_map in feat_maps.items():
            feat_map = nn.functional.interpolate(feat_map, size=self.out_size, mode=self.upsample)
            print(feat_map.shape)
            features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
        b, c, _ = features.shape
        features = torch.reshape(features, (b, c, self.out_size[0], self.out_size[1]))  # (1, 3456, 64, 64)
        return features

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = featureFusion()
    model = model.to(device)
    input = torch.randn(5, 3, 224, 224).to(device)
    out = model(input)

