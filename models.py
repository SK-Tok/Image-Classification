# import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models  as models

class ResNet50(nn.Module):
    def __init__(self, input_size, num_class=7):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained = True)
        self.feature = nn.Sequential(*list(resnet50.children())[:8])
        self.avgpool = nn.ArgPool2d(int(round(input_size / 32.0)))
        self.fc = nn.Linear(2048, num_class)
        self.fc.weight.data.normal_(0.001)
        self.fc.bias.data.zero_()
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, input_size, num_class=7):
        super(ResNet101, self).__init__()
        resnet101 = models.resnet101(pretrained = True)
        self.feature = nn.Sequential(*list(resnet101.children())[:8])
        self.avgpool = nn.ArgPool2d(int(round(input_size / 32.0)))
        self.fc = nn.Linear(2048, num_class)
        self.fc.weight.data.normal_(0.001)
        self.fc.bias.data.zero_()
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, input_size, num_class=7):
        super(ResNet152, self).__init__()
        resnet152 = models.resnet152(pretrained = True)
        self.feature = nn.Sequential(*list(resnet152.children())[:8])
        self.avgpool = nn.ArgPool2d(int(round(input_size / 32.0)))
        self.fc = nn.Linear(2048, num_class)
        self.fc.weight.data.normal_(0.001)
        self.fc.bias.data.zero_()
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, input_size, num_class=7):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained = True)
        vgg16.classifier[6] = nn.Linear(4096, num_class)
        vgg16.classifier[0] = nn.Linear((int(round(input_size / 32.0))**2)*512, 4096)
        self.feature = nn.Sequential(*list(vgg16.features))
        self.classifier = nn.Sequential(*list(vgg16.classifier))
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x
