import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConv2dNoReLU(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2dNoReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res

class Bottleneck(nn.Module):
    
    def __init__(self, in_planes, filters, stride, use_projection=False, padding=0):
        super(Bottleneck, self).__init__()
        self.use_projection = use_projection
        if use_projection:
            self.shortcut = BasicConv2dNoReLU(in_planes, filters*4, kernel_size=1, stride=stride, padding=0) # Padding = "SAME" if stride =1 else "VALID"
        self.conv1 = BasicConv2d(in_planes, filters, kernel_size=1, stride=1, padding=0)
        self.conv2 = BasicConv2d(filters, filters, kernel_size=3, stride=stride, padding=1)
        self.conv3 = BasicConv2dNoReLU(filters, 4 * filters, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.use_projection:
            out = self.add(out, self.shortcut(x))
        else:
            out = self.add(out,x)
        out = self.relu(out)
        return out


class Layer(nn.Module):
    def __init__(self, inputs, filters, blocks, stride, padding=0):
        super(Layer, self).__init__()
        layers = [None] * blocks
        layers[0] = Bottleneck(inputs, filters, stride, use_projection=True, padding=padding)
        for i in range(1, blocks):
            layers[i] = Bottleneck(filters*4, filters, 1, use_projection=False)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class ResnetTF(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000):
        if pretrained:
            print("Pretraining is really not an option for TF imports")
        super(ResnetTF, self).__init__()
        self.initial_conv = BasicConv2d(3, 64, kernel_size=7, stride=[2,2], padding=3) #Padding='same'
        self.pad = nn.ZeroPad2d((0,1,0,1))
        self.initial_max_pool = nn.MaxPool2d(3, stride=2, padding=0) # Padding = SAME
        self.layer1 = Layer(inputs=64, filters=64, blocks=3, stride=1, padding=1)
        self.layer2 = Layer(inputs=256, filters=128, blocks=4, stride=2, padding=1)
        self.layer3 = Layer(inputs=512, filters=256, blocks=6, stride=2, padding=1)
        self.layer4 = Layer(inputs=1024, filters=512, blocks=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.pad(x)
        x = self.initial_max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return(x)
