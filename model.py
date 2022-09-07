import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils

import ssl

model_name_ = "se_resnext101_32x4d"  # se_resnext50_32x4d

ssl._create_default_https_context = ssl._create_unverified_context


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CusModel(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CusModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.lin1 = nn.Linear(in_ch, 500)
        self.relu3 = nn.ReLU()

        self.lin2 = nn.Linear(500, out_ch)
        self.logSMax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)

        x = torch.flatten(x)
        x = self.lin1(x)
        x = self.relu3(x)

        x = self.lin2(x)
        x = self.logSMax(x)
        return x


def get_model(model_name, num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)

    # dim_feats = model.last_linear.in_features
    # model.last_linear = nn.Linear(dim_feats, num_classes)
    # model.avg_pool = nn.AdaptiveAvgPool2d(1) # 4.6..

    # freeze layer weights
    for param in model.parameters():
        param.requires_grad = False

    dim_feats = model.last_linear.in_features
    model.last_linear = CusModel(dim_feats, num_classes)

    model.last_linear = nn.Sequential(
        nn.Linear(dim_feats, 1024),
        nn.Linear(1024, num_classes)
    )
    return model


def main():
    model = get_model(model_name_)
    print(model)


if __name__ == '__main__':
    main()
