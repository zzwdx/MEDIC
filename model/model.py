# reference: https://github.com/thuml/OpenDG-DAML

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls
from torch.nn.parameter import Parameter

Parameter.fast = None

class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        if self.weight.fast is not None and self.bias.fast is not None:
            return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight.fast, self.bias.fast, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.conv3 = Conv2d_fw(in_channels=planes, out_channels=planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d_fw(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFast(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFast, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d_fw(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_fw(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_fw(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)

        self._out_features = 256

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)

        return x


class MutiClassifier(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.classifier = Linear_fw(feature_dim, self.num_classes)
        self.b_classifier = Linear_fw(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.classifier.weight, .1)
        nn.init.constant_(self.classifier.bias, 0.)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)
        x1 = self.classifier(x)
        x2 = self.b_classifier(x)
        return x1, x2


class MutiClassifier_(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier_, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.b_classifier = Linear_fw(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        x = x.view(x.size(0), 2, -1)
        x = x[:, 1, :]
            
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)   
        x2 = self.b_classifier(x)
        x1 = x2.view(x.size(0), 2, -1)
        x1 = x1[:, 1, :]
        return x1, x2


def resnet18_fast(progress=True):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters:
        - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
        - **progress** (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNetFast(BasicBlock, [2, 2, 2, 2])
    state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)
    del model.fc

    return model


def resnet50_fast(progress=True):
    model = ResNetFast(Bottleneck, [3, 4, 6, 3])
    state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)
    del model.fc

    return model

    


