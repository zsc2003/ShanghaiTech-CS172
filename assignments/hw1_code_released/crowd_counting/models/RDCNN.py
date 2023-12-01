import torch.nn as nn
from torchvision import models
import torch

class RDCNN(nn.Module):
    def __init__(self, pretained=False):
        super(RDCNN, self).__init__()
        resnet = models.resnet34(pretrained=pretained)
        self.Conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.Reslayer1 = resnet.layer1
        self.Reslayer2 = resnet.layer2
        self.Reslayer3 = resnet.layer3
        self.backend_feat = [256, 256, 256, 128, 64, 32]
        self.backend = make_layers(self.backend_feat, in_channels=256, batch_norm=True, dilation=True)
        self.output_layer = nn.Conv2d(32,1,1)

    def forward(self,x):
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.Reslayer1(x)
        x = self.Reslayer2(x)
        x = self.Reslayer3(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)

if __name__ == '__main__':
    device = 'cuda:0'
    input = torch.ones(size=(1, 3, 1024, 681)).to(device)
    model = RDCNN(pretained=True).to(device)
    output = model(input)
    print(output.size())




