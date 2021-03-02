import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from dataloader import UtilData, dataLoader
import torch.utils.data as data_utils
from torchvision import transforms

from feature_ae import FeatCAE


class AutoFeature(nn.Module):
    """
    CNN feature + Autoencoder loss
    """

    def __init__(self, args):
        super(AutoFeature, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size
        # feature extractor
        self.extractor = models.resnet50(pretrained=True).to(self.device)
        self.in_feat = args.dim_insVec
        self.latent_dim = args.dim_autoLatten

        self.autoencoder = FeatCAE(in_channels=self.in_feat, latent_dim=self.latent_dim, is_bn=True).to(self.device)


    def build_new_feature(self, input_data, label, train):
        # forward
        CNN_feature = self.extractor(input_data)


        if train:
            loss_list = []
            for i in range(self.batch_size):
                feat = CNN_feature[i:i + 1, :, :, :]
                print(feat.shape)
                exit()
                if label[i] == 0:
                    dec = self.autoencoder(feat)     # (1,2048,7,7)
                    feat = torch.flatten(feat, start_dim=2, end_dim=3)  # (1,2048,49)
                    dec = torch.flatten(dec, start_dim=2, end_dim=3)  # (1,2048,49)
                    # loss
                    self.loss = self.autoencoder.loss_function(dec, feat)   # (1, 49)
                    self.loss = torch.unsqueeze(self.loss, dim=1)                       # (1, 1, 49)
                else:
                    feat = torch.flatten(feat, start_dim=2, end_dim=3)  # (1,2048,49)
                    self.loss = torch.zeros(1, 1, feat.shape[2]).to(self.device)  # (1,1,49)
                loss_list.append(self.loss)

            self.loss = torch.cat(loss_list, dim=0)   # (5, 1,49)
            CNN_feature = torch.flatten(CNN_feature, start_dim=2, end_dim=3)  # (5,2048,49)

        if not train:
            dec = self.autoencoder(CNN_feature)
            CNN_feature = torch.flatten(CNN_feature, start_dim=2, end_dim=3)  # (5,2048,49)
            dec = torch.flatten(dec, start_dim=2, end_dim=3)  # (5,2048,49)
            # loss
            self.loss = self.autoencoder.loss_function(dec, CNN_feature)  # (5, 49)
            self.loss = torch.unsqueeze(self.loss, dim=1)  # (5, 1, 49)

        # contact
        newFeature = torch.cat([CNN_feature, self.loss], dim=1)
        return newFeature

    def forward(self, x, label, train):
        out = self.build_new_feature(x, label, train)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoFeature()
    model = model.to(device)
    input = torch.randn(5, 3, 224, 224).to(device)
    label = torch.ones(5)
    out = model(input, label, True)
    print(out.shape)



